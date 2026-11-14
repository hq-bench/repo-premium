#!/usr/bin/env python3
"""Retrieval SCORES for the post-retrieval (QPP) decision-model test.

The canonical caches keep ranked file lists but no scores, so score-based
query-performance predictors (top-1 score, gaps, NQC/WIG-style spread) were
never available to the router. This script re-runs, per repository, the two
pre-LLM retrievers of the v2 action set with the canonical runner's own
corpus, chunking, tokenization and pooling, and keeps the scores:

  bm25   BM25 over whole files (the tuned chunk size, c = 1e9)
  dense  jina-code-0.5b over c=1000 chunks, max-pooled per file

Output per repository: <out>/<repo>.jsonl with, per (cell, query_id), the
top-50 files and scores of each retriever plus corpus-wide score mean/std
(the NQC/WIG normalizers). Chunk embeddings are cached as .npy so later
stages (reranking the union) never re-encode.

Guards: BM25 top-10 must equal bm25_sweep202610 whole-file lists; dense
top-10 is compared with the cached CUDA run (canon202607) and the agreement
is reported, since MPS and CUDA arithmetic can break near-ties differently.
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import run_retrieval as S  # noqa: E402  canonical runner (cluster copy)

ROOT = _HERE.parent
DATA = DATA_ROOT
TOPN = 50
WHOLE = 1_000_000_000


def qrels_for(repo: str, qrel_roots: dict[str, Path]) -> list[tuple[str, dict]]:
    out = []
    for cell, root in qrel_roots.items():
        f = root / f"{repo}.jsonl"
        if f.is_file():
            for q in S.load_qrels(f, 300, 0, 3):  # canonical_sweep.slurm: --n 300
                q = dict(q)
                q["problem_statement"] = S.strip_pr_boilerplate(q["problem_statement"])
                out.append((cell, q))
    return out


def top(scores: np.ndarray, paths: list[str]) -> list[list]:
    order = np.argsort(-scores)[:TOPN]
    return [[paths[i], float(scores[i])] for i in order]


def run_repo(repo: str, clone: Path, qrel_roots, out_dir: Path, model, batch: int) -> None:
    dest = out_dir / f"{repo}.jsonl"
    if dest.is_file():
        print(f"[skip] {repo}", flush=True)
        return
    queries = qrels_for(repo, qrel_roots)
    paths = S.list_corpus(clone)
    texts = [S.read_file(clone, p) for p in paths]

    # BM25, whole file (one chunk per file)
    w_chunks, w_ranges = [], []
    for t in texts:
        cs = S.chunk_file(t, WHOLE, 10)
        w_ranges.append((len(w_chunks), len(w_chunks) + len(cs)))
        w_chunks.extend(cs)
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([S.tokenize(c) for c in w_chunks])

    # dense, c=1000 chunks, cached embeddings
    d_chunks, d_ranges = [], []
    for t in texts:
        cs = S.chunk_file(t, 1000, 10)
        d_ranges.append((len(d_chunks), len(d_chunks) + len(cs)))
        d_chunks.extend(cs)
    emb_f = out_dir / f"{repo}.emb.npy"
    if model is None:
        d_emb = None
    elif emb_f.is_file():
        d_emb = np.load(emb_f)
    else:
        d_emb = model.encode(d_chunks, batch_size=batch, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=True)
        np.save(emb_f, d_emb.astype(np.float32))

    rows = []
    for cell, q in queries:
        text = q["problem_statement"]
        cs = bm25.get_scores(S.tokenize(text))
        b = np.array([cs[s:e].max() if e > s else 0.0 for s, e in w_ranges])
        row = {"cell": cell, "query_id": q["query_id"], "gold_files": q["gold_files"],
               "bm25": top(b, paths), "bm25_mean": float(b.mean()), "bm25_std": float(b.std()),
               "n_files": len(paths)}
        if d_emb is not None:
            qe = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
            sims = d_emb @ qe
            d = np.array([sims[s:e].max() if e > s else -1.0 for s, e in d_ranges])
            row.update({"dense": top(d, paths), "dense_mean": float(d.mean()),
                        "dense_std": float(d.std())})
        rows.append(row)
    dest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"[done] {repo}: {len(rows)} queries, {len(d_chunks)} chunks", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clones", type=Path, required=True)
    ap.add_argument("--qrels-a", type=Path, required=True)
    ap.add_argument("--qrels-b", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DATA / "smoke_test_results/qpp_scores202610")
    ap.add_argument("--repos", nargs="*")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-dense", action="store_true", help="BM25 scores only (no encoder)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = None
    if not args.skip_dense:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("jinaai/jina-code-embeddings-0.5b", trust_remote_code=True,
                                    device=args.device)
    roots = {"A": args.qrels_a, "B": args.qrels_b}
    repos = args.repos or sorted(p.name for p in args.clones.iterdir() if p.is_dir())
    for repo in repos:
        run_repo(repo, args.clones / repo, roots, args.out, model, args.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
