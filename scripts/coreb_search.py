"""coreb_search.py — CLI retriever wrappers for agent-eval matrix.

Three modes:
  --mode grep   BM25 over file-chunks of the repo
  --mode dense  jina-code-embeddings-0.5b cosine over file-chunks
  --mode router tau-gated RRF blend of grep + dense

Two subcommands:
  build   Index a repo. Outputs {index_dir}/chunks.jsonl and {index_dir}/emb.npy.
  query   Query the pre-built index. Outputs JSON top-K results to stdout.

Chunking policy:
  - Only .py files are indexed.
  - File split into non-overlapping windows of ~1000 chars respecting line breaks.
  - Each chunk records (path, start_line, end_line, text).

Query output schema (JSON to stdout):
  { "hits": [ {"rank":1, "path":"a/b.py", "start_line":10, "end_line":42,
               "score":0.813, "snippet":"..."} , ...] }

Intended use inside mini-swe-agent: the agent invokes
  coreb_search --mode <mode> --index-dir /path/to/index --query "..." --k 20
and receives a JSON list of hits. Agent is instructed via system prompt to use
this tool instead of raw find/grep.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Lazy imports for speed at CLI parse
def _lazy_np():
    import numpy as np
    return np


CHUNK_CHARS = 1000
EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
                "build", "dist", ".tox", ".mypy_cache", ".pytest_cache",
                ".eggs", "site-packages"}


def iter_py_files(repo: Path):
    for p in repo.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.is_symlink():
            continue
        try:
            if p.stat().st_size > 500_000:
                continue
        except OSError:
            continue
        yield p


def chunk_file(rel_path: str, text: str, chunk_chars: int = CHUNK_CHARS):
    """Yield (start_line, end_line, chunk_text) windows respecting line boundaries."""
    lines = text.splitlines(keepends=True)
    buf = []
    buf_start = 1
    buf_len = 0
    line_no = 0
    for line in lines:
        line_no += 1
        if buf_len + len(line) > chunk_chars and buf:
            yield buf_start, line_no - 1, "".join(buf)
            buf = []
            buf_start = line_no
            buf_len = 0
        buf.append(line)
        buf_len += len(line)
    if buf:
        yield buf_start, line_no, "".join(buf)


def build_chunks(repo: Path):
    chunks = []
    for f in iter_py_files(repo):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(f.relative_to(repo))
        for s, e, t in chunk_file(rel, text):
            chunks.append({"path": rel, "start_line": s, "end_line": e, "text": t})
    return chunks


def cmd_build(args):
    repo = Path(args.repo).resolve()
    out = Path(args.index_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[build] repo={repo}", file=sys.stderr)
    chunks = build_chunks(repo)
    print(f"[build] {len(chunks)} chunks", file=sys.stderr)
    with (out / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    if args.mode in ("dense", "both"):
        np = _lazy_np()
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer(args.model, trust_remote_code=True)
        embs = model.encode(
            [c["text"] for c in chunks],
            batch_size=args.batch,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.save(out / "emb.npy", embs.astype("float32"))
        (out / "meta.json").write_text(json.dumps({
            "model": args.model, "n_chunks": len(chunks),
            "emb_dim": int(embs.shape[1]),
        }))
        print(f"[build] wrote emb.npy shape={embs.shape}", file=sys.stderr)


def load_chunks(index_dir: Path):
    return [json.loads(l) for l in (index_dir / "chunks.jsonl").read_text().splitlines() if l.strip()]


def snippet(text: str, max_chars: int = 300) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def bm25_scores(chunks, query: str):
    from rank_bm25 import BM25Okapi  # type: ignore
    import re
    tok = lambda s: re.findall(r"[A-Za-z_][A-Za-z_0-9]{2,}", s.lower())
    corpus = [tok(c["text"]) for c in chunks]
    bm = BM25Okapi(corpus)
    return bm.get_scores(tok(query))


def dense_scores(index_dir: Path, query: str, model_name: str):
    np = _lazy_np()
    from sentence_transformers import SentenceTransformer  # type: ignore
    embs = np.load(index_dir / "emb.npy")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    return embs @ q


def tau_query(query: str, chunks) -> float:
    """Deterministic top-K IDF proxy of stripped query, K=5, over the repo corpus.
    tau in [0,1] roughly; higher = more lexically distinctive."""
    import re, math
    from collections import Counter
    tok = lambda s: re.findall(r"[A-Za-z_][A-Za-z_0-9]{2,}", s.lower())
    df = Counter()
    for c in chunks:
        for t in set(tok(c["text"])):
            df[t] += 1
    N = max(len(chunks), 1)
    qtoks = tok(query)
    if not qtoks:
        return 0.0
    idfs = sorted(
        (math.log((N + 1) / (df.get(t, 0) + 1)) + 1 for t in set(qtoks)),
        reverse=True,
    )[:5]
    if not idfs:
        return 0.0
    return float(sum(idfs) / len(idfs)) / math.log(N + 1)


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> dict[int, float]:
    if weights is None:
        weights = [1.0] * len(rankings)
    score: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + w * 1.0 / (k + rank + 1)
    return score


def cmd_query(args):
    np = _lazy_np()
    index_dir = Path(args.index_dir).resolve()
    chunks = load_chunks(index_dir)
    if not chunks:
        print(json.dumps({"hits": [], "error": "empty index"}))
        return
    K = args.k

    if args.mode == "grep":
        scores = bm25_scores(chunks, args.query)
        order = np.argsort(-scores)[:K]
        top = [(int(i), float(scores[int(i)])) for i in order]
    elif args.mode == "dense":
        scores = dense_scores(index_dir, args.query, args.model)
        order = np.argsort(-scores)[:K]
        top = [(int(i), float(scores[int(i)])) for i in order]
    elif args.mode == "router":
        bm = bm25_scores(chunks, args.query)
        de = dense_scores(index_dir, args.query, args.model)
        tau = tau_query(args.query, chunks)
        # alpha in [0, 0.5]: high tau -> favor BM25 more
        alpha = 0.5 * max(0.0, min(1.0, (tau - 0.15) / 0.25))
        bm_rank = list(np.argsort(-bm))
        de_rank = list(np.argsort(-de))
        merged = rrf([de_rank, bm_rank], k=60, weights=[1.0 - alpha, alpha])
        order = sorted(merged, key=lambda i: -merged[i])[:K]
        top = [(int(i), float(merged[i])) for i in order]
    else:
        raise SystemExit(f"unknown mode: {args.mode}")

    hits = []
    for rank, (i, sc) in enumerate(top, 1):
        c = chunks[i]
        hits.append({
            "rank": rank,
            "path": c["path"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "score": round(sc, 4),
            "snippet": snippet(c["text"], args.snippet_chars),
        })
    print(json.dumps({"hits": hits}, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--repo", required=True)
    b.add_argument("--index-dir", required=True)
    b.add_argument("--mode", choices=["grep", "dense", "both"], default="both",
                   help="grep: chunks only (no embed). dense/both: embed too.")
    b.add_argument("--model", default="jinaai/jina-code-embeddings-0.5b")
    b.add_argument("--batch", type=int, default=64)
    b.set_defaults(func=cmd_build)

    q = sub.add_parser("query")
    q.add_argument("--index-dir", required=True)
    q.add_argument("--mode", choices=["grep", "dense", "router"], required=True)
    q.add_argument("--query", required=True)
    q.add_argument("--model", default="jinaai/jina-code-embeddings-0.5b")
    q.add_argument("--k", type=int, default=20)
    q.add_argument("--snippet-chars", type=int, default=300)
    q.set_defaults(func=cmd_query)

    args = ap.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
