"""
Phase 2.1 smoke test — django subset.

Validate the full grep + BM25 + dense retrieval pipeline + τ(q) computation
end-to-end on a small held-out subset of django Cell A qrels, before scaling
to non-Python repos and Cell B.

Steps:
  1. Load `--n` django qrels with all_files_exist=true and 1<=n_gold_files<=3.
  2. Build the corpus: every .py file under the pinned django snapshot.
  3. Run three retrievers over that corpus:
       - grep:  rank files by sum of IDF of query tokens that appear literally.
       - BM25:  rank_bm25.BM25Okapi over a code-aware tokenization.
       - dense: sentence-transformers model (default: jina-code-0.5b),
                cosine sim between problem_statement and per-file embedding.
  4. Per query: Recall@10, nDCG@10, τ(q) = mean IDF of top-5 rarest shared
     tokens between query and the first gold file.
  5. Embedding Premium Δ = nDCG_dense - nDCG_bm25; per-query + averaged.

Outputs:
  - `<out>/smoke_test_django.jsonl` (one record per query × retriever)
  - `<out>/smoke_test_django_summary.json` (aggregate metrics + Δ)
  - stdout: human-readable summary table.

Usage:
  python scripts/run_retrieval.py \
      --qrel-file ${REPO_PREMIUM_DATA:-data}/release/qrels/django__django.jsonl \
      --clone-dir ${REPO_PREMIUM_DATA:-data}/repos/django__django \
      --out-dir results/django \
      --n 20 \
      --dense-model jinaai/jina-embeddings-v2-base-code
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# camelCase boundaries: [a-z0-9]->[A-Z], or [A-Z]+->[A-Z][a-z] (URL|Path)
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
MAX_FILE_CHARS = 8000  # truncate per-file text for dense encoding

# Set from --tokenize-camelcase CLI at main() entry; keeps default (Python) semantics
# identical to prior paper runs when False.
TOKENIZE_CAMELCASE = False


def tokenize(text: str) -> list[str]:
    """Code-aware: identifiers + underscored words, lowercased.
    If TOKENIZE_CAMELCASE is set, additionally split camelCase identifiers
    into subwords (e.g. `getUserById` -> `get`, `User`, `By`, `Id`).
    """
    toks = TOKEN_RE.findall(text)
    if TOKENIZE_CAMELCASE:
        out: list[str] = []
        for t in toks:
            for part in t.split("_"):
                if part:
                    out.extend(s for s in _CAMEL_BOUNDARY_RE.split(part) if s)
        return [t.lower() for t in out]
    return [t.lower() for t in toks]


# Section headers that are boilerplate in common PR templates (django/sympy/sklearn/etc.)
_BOILERPLATE_HEADER_RE = re.compile(
    r"^\s*#+\s*("
    r"trac ticket|ai assistance|ai generation|checklist|reviewer|"
    r"how (to|has) [^\n]*been tested|related|fixes?|"
    r"other comments?|notes? to|open issues|prior conversation|"
    r"release notes?|"
    r"screenshot|screencast|please .* check|"
    r"reference issues|any other"
    r")",
    re.IGNORECASE,
)
# Token-style checkbox lines: "- [x] ...", "- [ ] ..."
_CHECKBOX_RE = re.compile(r"^\s*-\s*\[[ xX]\]")
_HEADER_RE = re.compile(r"^\s*#+\s+")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def strip_pr_boilerplate(text: str) -> str:
    """Remove common PR-template boilerplate while keeping the title + substantive prose.

    Strips: HTML comments, checkbox lines, and entire markdown sections under
    boilerplate-named headers (Trac ticket, AI Assistance, Checklist, ...).
    Kept-section headers are demoted (#### removed) so the text reads as prose.
    """
    text = _HTML_COMMENT_RE.sub("", text)
    out: list[str] = []
    skip = False
    for line in text.split("\n"):
        if _HEADER_RE.match(line):
            skip = bool(_BOILERPLATE_HEADER_RE.match(line))
            if not skip:
                out.append(_HEADER_RE.sub("", line))
            continue
        if skip:
            continue
        if _CHECKBOX_RE.match(line):
            continue
        out.append(line)
    result = "\n".join(out)
    result = re.sub(r"\n\s*\n+", "\n\n", result)
    return result.strip()


def chunk_file(text: str, target_chars: int = 1500,
               overlap_lines: int = 10) -> list[str]:
    """Sliding-window chunks: split by lines, ~target_chars each, with line overlap."""
    if not text.strip():
        return ["# empty file"]
    lines = text.split("\n")
    chunks: list[str] = []
    cur: list[str] = []
    cur_chars = 0
    for line in lines:
        cur.append(line)
        cur_chars += len(line) + 1
        if cur_chars >= target_chars:
            chunks.append("\n".join(cur))
            cur = cur[-overlap_lines:] if overlap_lines > 0 else []
            cur_chars = sum(len(l) + 1 for l in cur)
    if cur and "\n".join(cur).strip():
        chunks.append("\n".join(cur))
    return chunks or ["# empty file"]


def load_qrels(path: Path, n: int, seed: int = 0,
               max_gold_files: int = 3) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    # Cell B records carry a `filter_reason` field; None means passes quality filter.
    # Cell A records lack this field — .get(...) returns None → treated as pass.
    rows = [r for r in rows
            if r.get("all_files_exist")
            and 1 <= r["n_gold_files"] <= max_gold_files
            and not r.get("filter_reason")]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))[:n]
    return [rows[i] for i in idx]


def list_corpus(clone_dir: Path, exts: tuple[str, ...] = (".py",)) -> list[str]:
    """Relative paths of all files under clone_dir matching any of `exts`."""
    exts_norm = tuple(e if e.startswith(".") else "." + e for e in exts)
    paths = sorted({
        p.relative_to(clone_dir).as_posix()
        for e in exts_norm
        for p in clone_dir.rglob(f"*{e}")
    })
    return paths


def read_file(clone_dir: Path, rel: str) -> str:
    try:
        return (clone_dir / rel).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def compute_idf(doc_tokens: list[list[str]]) -> dict[str, float]:
    N = len(doc_tokens)
    df: Counter = Counter()
    for toks in doc_tokens:
        df.update(set(toks))
    return {t: math.log((N + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def grep_rank(query_tokens: list[str], chunk_tokens: list[list[str]],
              file_chunk_ranges: list[tuple[int, int]], paths: list[str],
              idf: dict[str, float], k: int = 10) -> list[str]:
    """Chunk-level grep (IDF-sum of shared tokens), max-pooled per file."""
    qset = set(query_tokens)
    chunk_scores = np.array([
        sum(idf.get(t, 0.0) for t in (qset & set(toks)))
        for toks in chunk_tokens
    ])
    file_scores = np.array([
        float(chunk_scores[s:e].max()) if e > s else 0.0
        for s, e in file_chunk_ranges
    ])
    order = np.argsort(-file_scores)[:k]
    return [paths[i] for i in order]


def bm25_rank(query_tokens: list[str], chunk_tokens: list[list[str]],
              file_chunk_ranges: list[tuple[int, int]], paths: list[str],
              bm25: Any, k: int = 10) -> list[str]:
    """Chunk-level BM25, max-pooled per file. Pass a prefit BM25Okapi instance."""
    chunk_scores = bm25.get_scores(query_tokens)
    file_scores = np.array([
        float(chunk_scores[s:e].max()) if e > s else 0.0
        for s, e in file_chunk_ranges
    ])
    order = np.argsort(-file_scores)[:k]
    return [paths[i] for i in order]


def dense_rank(query: str, paths: list[str], texts: list[str],
               model_name: str, k: int = 10,
               batch_size: int = 16) -> list[str]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, trust_remote_code=True)
    q_emb = model.encode([query], normalize_embeddings=True,
                         convert_to_numpy=True)[0]
    chunks = [t[:MAX_FILE_CHARS] for t in texts]
    d_emb = model.encode(chunks, batch_size=batch_size,
                         normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=True)
    sims = d_emb @ q_emb
    order = np.argsort(-sims)[:k]
    return [paths[i] for i in order]


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / max(len(gold), 1)


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    for i, p in enumerate(ranked[:k]):
        if p in gold:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg > 0 else 0.0


def tau_q(query_tokens: list[str], gold_tokens: list[str],
          idf: dict[str, float], top_k: int = 5) -> float:
    """Mean IDF of the top-K rarest tokens shared between query and gold file."""
    shared = set(query_tokens) & set(gold_tokens)
    if not shared:
        return 0.0
    ranked = sorted(shared, key=lambda t: -idf.get(t, 0.0))[:top_k]
    vals = [idf.get(t, 0.0) for t in ranked]
    return sum(vals) / len(vals) if vals else 0.0


@dataclass
class QueryResult:
    query_id: str
    gold_files: list[str]
    tau_q: float
    grep_top10: list[str]
    bm25_top10: list[str]
    dense_top10: list[str]
    grep_ndcg10: float
    bm25_ndcg10: float
    dense_ndcg10: float
    grep_recall10: float
    bm25_recall10: float
    dense_recall10: float
    delta_ndcg10: float  # dense - bm25
    # reranker fields (defaults so old paths still work if reranker off)
    rerank_top10: list[str] = None
    rerank_ndcg10: float = -1.0
    rerank_recall10: float = -1.0
    delta_rerank_ndcg10: float = 0.0   # rerank - dense



def _load_reranker(model_id: str, task: str):
    """Load hq-bench/coreb-code-reranker style CausalLM cross-encoder.

    Returns (score_fn, cleanup). score_fn takes (query, document) -> float.
    Uses logit(yes)-logit(no) at last position.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    task_map = {
        "text2code": "Given a natural language programming task, retrieve code that correctly solves or implements the task.",
        "code2code": "Given a code snippet, retrieve code that is semantically equivalent or solves the same task.",
        "code2text": "Given a code snippet, retrieve the natural language description or problem statement that best matches the code.",
    }
    instruct = task_map[task]

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    yes_id = tok.convert_tokens_to_ids("yes")
    no_id = tok.convert_tokens_to_ids("no")
    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements "
              "based on the Query and the Instruct provided. Note that the answer "
              "can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

    @torch.no_grad()
    def score(query: str, document: str) -> float:
        prompt = f"{PREFIX}<Instruct>: {instruct}\n<Query>: {query}\n<Document>: {document}{SUFFIX}"
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        logits = model(**inputs).logits[0, -1, :]
        return float((logits[yes_id] - logits[no_id]).item())

    return score

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qrel-file", type=Path, required=True)
    p.add_argument("--clone-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dense-model", default="jinaai/jina-code-embeddings-0.5b")
    p.add_argument("--max-gold-files", type=int, default=3,
                   help="Maximum gold files per retained query (default preserves paper runs)")
    p.add_argument("--encode-batch", type=int, default=16,
                   help="Batch size for dense encoding (lower for larger models / OOM safety)")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--chunk-chars", type=int, default=1500,
                   help="Target chars per chunk (sliding-window)")
    p.add_argument("--chunk-overlap-lines", type=int, default=10,
                   help="Lines of overlap between adjacent chunks")
    p.add_argument("--max-seq-length", type=int, default=0,
                   help="If >0, cap model.max_seq_length to this many tokens "
                        "(mitigates SDPA OOM on pathological long-line chunks; "
                        "0 = use model default).")
    p.add_argument("--strip-boilerplate", action="store_true", default=True,
                   help="Strip PR-template boilerplate before retrieval (default: on)")
    p.add_argument("--no-strip-boilerplate", dest="strip_boilerplate", action="store_false",
                   help="Disable PR-template boilerplate stripping")
    p.add_argument("--code-exts", default=".py",
                   help="Comma-separated file extensions to include in the corpus "
                        "(default: .py). Example for JS: '.js,.ts,.tsx,.jsx,.vue,.mjs'.")
    p.add_argument("--tokenize-camelcase", action="store_true",
                   help="Split camelCase identifiers into subwords in tokenize() "
                       "(for JS/TS/Java/Go corpora). Off by default: preserves "
                       "existing Python paper numbers.")
    p.add_argument("--query-field", default="problem_statement",
                   help="Qrel field to use as the query text. Lets an "
                        "LLM-expanded variant be evaluated against the same "
                        "corpus, qrels and chunking as the original query.")
    p.add_argument("--skip-dense", action="store_true",
                   help="Skip loading and running the dense encoder. Dense "
                        "metrics are emitted as -1.0 sentinels. Used by the "
                        "generative-lexical arm, where dense on the original "
                        "query is already cached and only BM25/grep need "
                        "recomputing on the expanded query.")
    p.add_argument("--rerank-model", default="",
                   help="HF model id for cross-encoder reranker (e.g. hq-bench/coreb-code-reranker). "
                        "If empty, no reranking is performed.")
    p.add_argument("--rerank-topk", type=int, default=50,
                   help="Number of dense-top candidates to feed into reranker.")
    p.add_argument("--rerank-task", default="text2code",
                   choices=["text2code", "code2code", "code2text"],
                   help="Task template for the reranker prompt (coreb-code-reranker convention).")
    args = p.parse_args()

    # Wire tokenizer mode BEFORE any tokenize() call happens
    if args.tokenize_camelcase:
        globals()["TOKENIZE_CAMELCASE"] = True

    exts = tuple(e.strip() for e in args.code_exts.split(",") if e.strip())

    rerank_score = None
    if args.rerank_model:
        print(f"[rerank] loading {args.rerank_model} (task={args.rerank_task}, topk={args.rerank_topk})", file=sys.stderr)
        rerank_score = _load_reranker(args.rerank_model, args.rerank_task)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] qrels from {args.qrel_file}", file=sys.stderr)
    qrels = load_qrels(args.qrel_file, args.n, args.seed, args.max_gold_files)
    print(f"[load] {len(qrels)} queries", file=sys.stderr)

    if args.query_field != "problem_statement":
        missing = [q["query_id"] for q in qrels if not q.get(args.query_field)]
        if missing:
            raise SystemExit(
                f"--query-field {args.query_field!r} is empty or absent for "
                f"{len(missing)} queries, e.g. {missing[:3]}"
            )
        for q in qrels:
            q["problem_statement"] = q[args.query_field]
        print(f"[query] using field {args.query_field!r} as the query text",
              file=sys.stderr)

    if args.strip_boilerplate:
        n_before = sum(len(q["problem_statement"]) for q in qrels)
        for q in qrels:
            q["problem_statement"] = strip_pr_boilerplate(q["problem_statement"])
        n_after = sum(len(q["problem_statement"]) for q in qrels)
        print(f"[strip] PR boilerplate stripped: {n_before} -> {n_after} chars "
              f"({100*(n_before-n_after)/max(n_before,1):.1f}% removed)",
              file=sys.stderr)

    print(f"[corpus] listing {exts} files under {args.clone_dir}", file=sys.stderr)
    paths = list_corpus(args.clone_dir, exts)
    print(f"[corpus] {len(paths)} files", file=sys.stderr)

    # Pre-read + chunk + tokenize once (chunk-level retrieval, max-pool per file)
    print(f"[corpus] reading + chunking (~{args.chunk_chars} chars/chunk, "
          f"{args.chunk_overlap_lines}-line overlap)...", file=sys.stderr)
    texts = [read_file(args.clone_dir, p) for p in paths]
    chunk_texts: list[str] = []
    file_chunk_ranges: list[tuple[int, int]] = []
    for t in texts:
        cs = chunk_file(t, args.chunk_chars, args.chunk_overlap_lines)
        start = len(chunk_texts)
        chunk_texts.extend(cs)
        file_chunk_ranges.append((start, start + len(cs)))
    avg_chunks = len(chunk_texts) / max(len(paths), 1)
    print(f"[corpus] {len(chunk_texts)} chunks across {len(paths)} files "
          f"(avg {avg_chunks:.1f}/file)", file=sys.stderr)
    chunk_tokens = [tokenize(c) for c in chunk_texts]

    # Sanity check: all gold files must be in the corpus
    path_set = set(paths)
    for q in qrels:
        missing_from_corpus = [g for g in q["gold_files"] if g not in path_set]
        if missing_from_corpus:
            print(f"[warn] {q['query_id']} gold not in corpus: {missing_from_corpus}",
                  file=sys.stderr)

    print(f"[idf] computing chunk-level IDF...", file=sys.stderr)
    idf = compute_idf(chunk_tokens)

    print(f"[bm25] fitting BM25Okapi over {len(chunk_tokens)} chunks...", file=sys.stderr)
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(chunk_tokens)

    # Run dense once for all queries (load model once, batch encode chunks once)
    model = None
    d_emb = None
    if args.skip_dense:
        print("[dense] skipped (--skip-dense); dense metrics emit -1.0",
              file=sys.stderr)
    else:
        print(f"[dense] encoding {len(chunk_texts)} chunks with {args.dense_model}",
              file=sys.stderr)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.dense_model, trust_remote_code=True)
        if args.max_seq_length > 0:
            print(f"[dense] capping max_seq_length: {model.max_seq_length} -> "
                  f"{args.max_seq_length}", file=sys.stderr)
            model.max_seq_length = args.max_seq_length
        d_emb = model.encode(chunk_texts, batch_size=args.encode_batch,
                             normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=True)

    results: list[QueryResult] = []
    for q in qrels:
        qid = q["query_id"]
        query = q["problem_statement"]
        q_toks = tokenize(query)
        gold = set(q["gold_files"])

        # τ(q): use union of token sets across chunks of first gold file
        first_gold = q["gold_files"][0]
        if first_gold in path_set:
            gidx = paths.index(first_gold)
            gs, ge = file_chunk_ranges[gidx]
            gold_toks: list[str] = []
            for ci in range(gs, ge):
                gold_toks.extend(chunk_tokens[ci])
            tau = tau_q(q_toks, gold_toks, idf, top_k=5)
        else:
            tau = 0.0

        grep_top = grep_rank(q_toks, chunk_tokens, file_chunk_ranges, paths,
                             idf, k=args.k)
        bm25_top = bm25_rank(q_toks, chunk_tokens, file_chunk_ranges, paths,
                             bm25, k=args.k)

        if args.skip_dense:
            file_sims = None
            dense_top = []
        else:
            q_emb = model.encode([query], normalize_embeddings=True,
                                 convert_to_numpy=True)[0]
            chunk_sims = d_emb @ q_emb
            file_sims = np.array([
                float(chunk_sims[s:e].max()) if e > s else -1.0
                for s, e in file_chunk_ranges
            ])
            order = np.argsort(-file_sims)[:args.k]
            dense_top = [paths[i] for i in order]

        # Rerank if enabled: take dense-top-K candidates, rerank, take top args.k
        rerank_top: list[str] = []
        rerank_ndcg = -1.0
        rerank_recall = -1.0
        if rerank_score is not None and file_sims is not None:
            # dense-top-K candidate FILES (not chunks) — reuse file_sims order
            cand_order = np.argsort(-file_sims)[:args.rerank_topk]
            cand_paths = [paths[i] for i in cand_order]
            # Score each candidate: document = full file text (truncated by tokenizer)
            scores = []
            for cp in cand_paths:
                doc = read_file(args.clone_dir, cp)[:MAX_FILE_CHARS]
                scores.append(rerank_score(query, doc))
            rr_order = np.argsort(-np.asarray(scores))[:args.k]
            rerank_top = [cand_paths[j] for j in rr_order]
            rerank_ndcg = ndcg_at_k(rerank_top, gold, args.k)
            rerank_recall = recall_at_k(rerank_top, gold, args.k)

        grep_ndcg = ndcg_at_k(grep_top, gold, args.k)
        bm25_ndcg = ndcg_at_k(bm25_top, gold, args.k)
        # -1.0 sentinel rather than 0.0: a skipped encoder must never be
        # mistaken downstream for an encoder that scored nothing.
        dense_ndcg = -1.0 if args.skip_dense else ndcg_at_k(dense_top, gold, args.k)

        r = QueryResult(
            query_id=qid,
            gold_files=q["gold_files"],
            tau_q=tau,
            grep_top10=grep_top,
            bm25_top10=bm25_top,
            dense_top10=dense_top,
            grep_ndcg10=grep_ndcg,
            bm25_ndcg10=bm25_ndcg,
            dense_ndcg10=dense_ndcg,
            grep_recall10=recall_at_k(grep_top, gold, args.k),
            bm25_recall10=recall_at_k(bm25_top, gold, args.k),
            dense_recall10=-1.0 if args.skip_dense else recall_at_k(dense_top, gold, args.k),
            delta_ndcg10=0.0 if args.skip_dense else dense_ndcg - bm25_ndcg,
            rerank_top10=rerank_top,
            rerank_ndcg10=rerank_ndcg,
            rerank_recall10=rerank_recall,
            delta_rerank_ndcg10=(rerank_ndcg - dense_ndcg) if rerank_score is not None else 0.0,
        )
        results.append(r)
        print(f"  [{qid:40s}]  τ={tau:5.2f}  "
              f"nDCG grep={grep_ndcg:.3f} bm25={bm25_ndcg:.3f} "
              f"dense={dense_ndcg:.3f}  Δ={r.delta_ndcg10:+.3f}",
              file=sys.stderr)

    # Write per-query JSONL
    out_jsonl = args.out_dir / "smoke_test_django.jsonl"
    with out_jsonl.open("w") as f:
        for r in results:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")

    # Summary
    n = len(results)
    def mean(xs: list[float]) -> float:
        return sum(xs) / n if n else 0.0

    summary: dict[str, Any] = {
        "n_queries": n,
        "k": args.k,
        "dense_model": args.dense_model,
        "n_corpus_files": len(paths),
        "mean_tau_q": mean([r.tau_q for r in results]),
        "ndcg10": {
            "grep": mean([r.grep_ndcg10 for r in results]),
            "bm25": mean([r.bm25_ndcg10 for r in results]),
            "dense": mean([r.dense_ndcg10 for r in results]),
            "premium_dense_minus_bm25": mean([r.delta_ndcg10 for r in results]),
            "premium_dense_minus_grep": mean(
                [r.dense_ndcg10 - r.grep_ndcg10 for r in results]),
        },
        "recall10": {
            "grep": mean([r.grep_recall10 for r in results]),
            "bm25": mean([r.bm25_recall10 for r in results]),
            "dense": mean([r.dense_recall10 for r in results]),
        },
    }
    if any(r.rerank_ndcg10 >= 0 for r in results):
        rr = [r for r in results if r.rerank_ndcg10 >= 0]
        summary["rerank_model"] = args.rerank_model
        summary["rerank_topk"] = args.rerank_topk
        summary["ndcg10"]["rerank"] = sum(r.rerank_ndcg10 for r in rr) / len(rr)
        summary["ndcg10"]["premium_rerank_minus_dense"] = sum(r.delta_rerank_ndcg10 for r in rr) / len(rr)
        summary["recall10"]["rerank"] = sum(r.rerank_recall10 for r in rr) / len(rr)

    print("\n=== SMOKE TEST SUMMARY ===", file=sys.stderr)
    print(f"n={n} queries  corpus={len(paths)} files ({','.join(exts)})  k={args.k}",
          file=sys.stderr)
    print(f"mean τ(q) = {summary['mean_tau_q']:.3f}", file=sys.stderr)
    print(f"{'retriever':<10s}  {'nDCG@10':>8s}  {'Recall@10':>10s}",
          file=sys.stderr)
    row_names = ["grep", "bm25", "dense"]
    if "rerank" in summary["ndcg10"]:
        row_names.append("rerank")
    for name in row_names:
        print(f"  {name:<8s}  {summary['ndcg10'][name]:>8.3f}  "
              f"{summary['recall10'][name]:>10.3f}", file=sys.stderr)
    print(f"\nEmbedding Premium Δ(dense - bm25)  = "
          f"{summary['ndcg10']['premium_dense_minus_bm25']:+.4f}",
          file=sys.stderr)
    print(f"Embedding Premium Δ(dense - grep)  = "
          f"{summary['ndcg10']['premium_dense_minus_grep']:+.4f}",
          file=sys.stderr)

    summary_path = args.out_dir / "smoke_test_django_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[written] {out_jsonl}", file=sys.stderr)
    print(f"[written] {summary_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
