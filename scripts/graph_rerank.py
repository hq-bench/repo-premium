"""R6 Step 3b — graph-augmented rerank of dense top-K.

For each query, takes the existing dense_top10 file ranking and reranks within
it using a simple graph-centrality signal: a file's score is boosted by how
many *other* files in the same top-K are in its 1-hop graph neighborhood.

Combined score:
  score(f) = (K - dense_rank(f)) + α * |neighbors(f) ∩ top_K \\ {f}|

where neighbors(f) = files reachable via outgoing/incoming calls or inherits
(GraphIndex.neighborhood, depth=1, no siblings).

This tests whether the dense top-10 already contains the *right cluster* of
files but ranks the non-central one first — a hypothesis where graph is a
useful tie-breaker on top of dense, not a replacement for it.

Output (--out-file): per-query JSONL
    {query_id, gold_files,
     dense_top10, dense_ndcg10, dense_recall10,
     graph_rerank_top10, graph_rerank_ndcg10, graph_rerank_recall10,
     n_graph_boosted}

Usage:
  python scripts/graph_rerank.py \\
      --dense-jsonl ${REPO_PREMIUM_DATA:-data}/coreb_repo_jsonl/cellA_c1000/django__django_c1000/smoke_test_django.jsonl \\
      --graph-jsonl ${REPO_PREMIUM_DATA:-data}/graph/django__django/graph.jsonl \\
      --out-file ${REPO_PREMIUM_DATA:-data}/coreb_repo_graph_rerank/cellA/jina-0.5b/django__django.jsonl \\
      --alpha 1.5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Import the existing GraphIndex.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_expand import GraphIndex  # noqa: E402


def dcg(rels: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked: list[str], gold: set[str], k: int = 10) -> float:
    rels = [1.0 if f in gold else 0.0 for f in ranked[:k]]
    ideal = [1.0] * min(k, len(gold)) + [0.0] * max(0, k - len(gold))
    d = dcg(rels)
    idl = dcg(ideal)
    return d / idl if idl > 0 else 0.0


def recall_at_k(ranked: list[str], gold: set[str], k: int = 10) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def graph_rerank(
    top_k: list[str],
    g: GraphIndex,
    alpha: float,
) -> tuple[list[str], int]:
    """Rerank top_k by combined dense-rank + graph-centrality score.

    Returns (reranked_files, n_files_boosted).
    """
    k = len(top_k)
    # Precompute 1-hop neighborhood of each file in top_k.
    nbrs = {f: g.neighborhood(f, depth=1, include_siblings=False) for f in top_k}
    top_set = set(top_k)
    rank_pos = {f: i for i, f in enumerate(top_k)}
    scores: list[tuple[float, int, str]] = []
    n_boosted = 0
    for f in top_k:
        graph_score = len(nbrs[f] & (top_set - {f}))
        if graph_score > 0:
            n_boosted += 1
        combined = (k - rank_pos[f]) + alpha * graph_score
        # Tie-break: prefer the original dense rank (lower index).
        scores.append((-combined, rank_pos[f], f))
    scores.sort()
    return [f for _, _, f in scores], n_boosted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dense-jsonl", type=Path, required=True)
    ap.add_argument("--graph-jsonl", type=Path, required=True)
    ap.add_argument("--out-file", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=1.5,
                    help="Graph-centrality weight in combined score.")
    args = ap.parse_args()

    args.out_file.parent.mkdir(parents=True, exist_ok=True)

    g = GraphIndex.from_jsonl(args.graph_jsonl)
    print(f"[rerank] loaded graph: {len(g.files)} files", file=sys.stderr)

    dense_rows = [json.loads(l) for l in open(args.dense_jsonl)]
    print(f"[rerank] {len(dense_rows)} queries from {args.dense_jsonl}",
          file=sys.stderr)

    out_lines: list[str] = []
    n_changed = 0
    n_better = 0
    n_worse = 0
    n_boosted_total = 0
    nd_dense_sum = 0.0
    nd_rerank_sum = 0.0
    for r in dense_rows:
        gold = set(r["gold_files"])
        top10 = r["dense_top10"]
        reranked, n_boosted = graph_rerank(top10, g, args.alpha)
        nd_d = ndcg_at_k(top10, gold, k=10)
        nd_r = ndcg_at_k(reranked, gold, k=10)
        rc_d = recall_at_k(top10, gold, k=10)
        rc_r = recall_at_k(reranked, gold, k=10)
        nd_dense_sum += nd_d
        nd_rerank_sum += nd_r
        n_boosted_total += n_boosted
        if reranked != top10:
            n_changed += 1
            if nd_r > nd_d:
                n_better += 1
            elif nd_r < nd_d:
                n_worse += 1
        out_lines.append(json.dumps({
            "query_id": r["query_id"],
            "gold_files": r["gold_files"],
            "dense_top10": top10,
            "dense_ndcg10": nd_d,
            "dense_recall10": rc_d,
            "graph_rerank_top10": reranked,
            "graph_rerank_ndcg10": nd_r,
            "graph_rerank_recall10": rc_r,
            "n_graph_boosted": n_boosted,
        }))
    with args.out_file.open("w") as f:
        f.write("\n".join(out_lines) + "\n")

    n = max(len(dense_rows), 1)
    print(f"[rerank] wrote {args.out_file}", file=sys.stderr)
    print(f"  queries           = {len(dense_rows)}", file=sys.stderr)
    print(f"  changed rank      = {n_changed} ({n_changed/n*100:.1f}%)",
          file=sys.stderr)
    print(f"    better          = {n_better}", file=sys.stderr)
    print(f"    worse           = {n_worse}", file=sys.stderr)
    print(f"  avg files boosted = {n_boosted_total/n:.2f} / 10",
          file=sys.stderr)
    print(f"  dense  nDCG@10    = {nd_dense_sum/n:.4f}", file=sys.stderr)
    print(f"  rerank nDCG@10    = {nd_rerank_sum/n:.4f}", file=sys.stderr)
    print(f"  delta             = {(nd_rerank_sum-nd_dense_sum)/n:+.4f}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
