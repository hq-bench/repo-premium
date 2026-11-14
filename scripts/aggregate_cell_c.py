"""Aggregate Cell C eval results → Finding 17 table.

Reads the per-(repo, encoder) outputs produced by cell_c_eval.slurm:
  smoke_test_results/cellC_<encoder>_c1000/<repo>/per_query_{grep,bm25,dense}.jsonl
  coreb_repo_graph_cellC/<repo>.jsonl              (pure graph)
  coreb_repo_graph_rerank/cellC/<encoder>/<repo>.jsonl  (dense + graph rerank)

And the original-query baseline:
  ${REPO_PREMIUM_DATA:-data}/r7_strat_pooled.jsonl  (built earlier; carries per-retriever nDCG@10
                               keyed by `_key = "<cell>__<source_query_id>"`)

For each query that has both a rewritten and an original score, computes:
  Δ_retriever = nDCG@10_rewritten − nDCG@10_original

Per-retriever pooled means + per-encoder means + per-cell means + Δ vs the
F8 prediction (BM25 −0.20, graph −0.45, dense −0.05 to −0.10).

Usage:
  python scripts/aggregate_cell_c.py \
      --results-root ${REPO_PREMIUM_DATA:-data}/coreb_repo_cellC_results \
      --strat-pooled ${REPO_PREMIUM_DATA:-data}/r7_strat_pooled.jsonl \
      --cell-c       ${REPO_PREMIUM_DATA:-data}/coreb_repo_cell_c.jsonl \
      --out          ${REPO_PREMIUM_DATA:-data}/coreb_repo_cellC_aggregated.json
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ENCODERS = ["jina-0.5b", "jina-1.5b", "Qwen3-0.6B", "Qwen3-4B"]
REPOS = [
    "django__django", "sympy__sympy", "sphinx-doc__sphinx",
    "matplotlib__matplotlib", "scikit-learn__scikit-learn",
    "astropy__astropy", "pydata__xarray", "pytest-dev__pytest",
    "pylint-dev__pylint", "psf__requests",
]


def repo_to_filename(repo: str) -> str:
    return repo.replace("/", "__")


def safe_load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def load_smoke(root: Path, encoder: str, repo: str) -> dict[str, dict]:
    """Return {qid: {grep, bm25, dense}} parsed from smoke_test_django.jsonl."""
    p = root / f"cellC_{encoder}_c1000" / repo / "smoke_test_django.jsonl"
    out: dict[str, dict] = {}
    for r in safe_load_jsonl(p):
        qid = r["query_id"]
        out[qid] = {
            "grep": r.get("grep_ndcg10"),
            "bm25": r.get("bm25_ndcg10"),
            "dense": r.get("dense_ndcg10"),
        }
    return out


def load_graph(graph_root: Path, repo: str) -> dict[str, float]:
    p = graph_root / f"{repo}.jsonl"
    return {r["query_id"]: r["graph_ndcg10"] for r in safe_load_jsonl(p)}


def load_rerank(rerank_root: Path, encoder: str, repo: str) -> dict[str, float]:
    p = rerank_root / "cellC" / encoder / f"{repo}.jsonl"
    return {r["query_id"]: r["graph_rerank_ndcg10"] for r in safe_load_jsonl(p)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path,
                    default=(DATA_ROOT / 'coreb_repo_cellC_results/smoke_test_results'))
    ap.add_argument("--graph-root", type=Path,
                    default=(DATA_ROOT / 'coreb_repo_cellC_results/coreb_repo_graph_cellC'))
    ap.add_argument("--rerank-root", type=Path,
                    default=(DATA_ROOT / 'coreb_repo_cellC_results/coreb_repo_graph_rerank'))
    ap.add_argument("--strat-pooled", type=Path,
                    default=(DATA_ROOT / 'r7_strat_pooled.jsonl'))
    ap.add_argument("--cell-c", type=Path,
                    default=(DATA_ROOT / 'coreb_repo_cell_c.jsonl'))
    ap.add_argument("--out", type=Path,
                    default=(DATA_ROOT / 'coreb_repo_cellC_aggregated.json'))
    args = ap.parse_args()

    # 1. Index original-query baseline.
    orig_by_source: dict[str, dict] = {}
    for line in args.strat_pooled.open():
        q = json.loads(line)
        # _key = "<cell>__<source_query_id>"
        source = q["_key"].split("__", 1)[1]
        orig_by_source[source] = q

    # 2. Index Cell C → source_query_id, cell, repo.
    cell_c_meta: dict[str, dict] = {}
    for line in args.cell_c.open():
        r = json.loads(line)
        cell_c_meta[r["query_id"]] = r

    # 3. Collect per-query rewritten scores across all (repo, encoder).
    # per_query[qid][encoder][retriever] = ndcg10
    per_query: dict[str, dict] = defaultdict(lambda: defaultdict(dict))
    # Encoder-agnostic: bm25, grep, graph
    for repo in REPOS:
        graph_scores = load_graph(args.graph_root, repo)
        for qid, s in graph_scores.items():
            per_query[qid]["__shared__"]["graph"] = s
        for encoder in ENCODERS:
            smoke = load_smoke(args.results_root, encoder, repo)
            rerank = load_rerank(args.rerank_root, encoder, repo)
            for qid, scores in smoke.items():
                per_query[qid][encoder]["bm25"] = scores.get("bm25")
                per_query[qid][encoder]["grep"] = scores.get("grep")
                per_query[qid][encoder]["dense"] = scores.get("dense")
            for qid, s in rerank.items():
                per_query[qid][encoder]["rerank"] = s

    # 4. Pooled means per retriever + Δ vs original.
    pooled: dict[str, dict] = {}
    for retr in ("bm25", "grep"):
        # Use first encoder's bm25/grep (deterministic, same across encoders).
        rewrites = []
        origs = []
        for qid, by_enc in per_query.items():
            v = by_enc.get(ENCODERS[0], {}).get(retr)
            if v is None:
                continue
            src = cell_c_meta.get(qid, {}).get("source_query_id")
            if not src or src not in orig_by_source:
                continue
            rewrites.append(v)
            origs.append(orig_by_source[src].get(retr))
        if rewrites:
            pooled[retr] = {
                "n": len(rewrites),
                "rewritten": mean(rewrites),
                "original": mean([o for o in origs if o is not None]),
                "delta": mean([r - o for r, o in zip(rewrites, origs) if o is not None]),
            }

    # graph (encoder-agnostic).
    graph_rewrites, graph_origs = [], []
    for qid, by_enc in per_query.items():
        v = by_enc.get("__shared__", {}).get("graph")
        if v is None:
            continue
        src = cell_c_meta.get(qid, {}).get("source_query_id")
        if not src or src not in orig_by_source:
            continue
        graph_rewrites.append(v)
        graph_origs.append(orig_by_source[src].get("graph"))
    if graph_rewrites:
        pooled["graph"] = {
            "n": len(graph_rewrites),
            "rewritten": mean(graph_rewrites),
            "original": mean([o for o in graph_origs if o is not None]),
            "delta": mean([r - o for r, o in zip(graph_rewrites, graph_origs)
                           if o is not None]),
        }

    # dense + rerank, per encoder.
    for encoder in ENCODERS:
        for retr in ("dense", "rerank"):
            rewrites, origs = [], []
            for qid, by_enc in per_query.items():
                v = by_enc.get(encoder, {}).get(retr)
                if v is None:
                    continue
                src = cell_c_meta.get(qid, {}).get("source_query_id")
                if not src or src not in orig_by_source:
                    continue
                rewrites.append(v)
                key_orig = (f"dense_{encoder}" if retr == "dense"
                            else f"rerank_{encoder}")
                origs.append(orig_by_source[src].get(key_orig))
            if rewrites:
                pooled[f"{retr}_{encoder}"] = {
                    "n": len(rewrites),
                    "rewritten": mean(rewrites),
                    "original": mean([o for o in origs if o is not None]),
                    "delta": mean([r - o for r, o in zip(rewrites, origs)
                                   if o is not None]),
                }

    # 5. By cell (A vs B) — descriptive vs already-descriptive sanity.
    by_cell = {"A": defaultdict(list), "B": defaultdict(list)}
    for qid, by_enc in per_query.items():
        meta = cell_c_meta.get(qid)
        if not meta:
            continue
        cell = meta["cell_origin"]
        for retr, score in by_enc.get(ENCODERS[0], {}).items():
            if retr in ("bm25", "grep") and score is not None:
                by_cell[cell][retr].append(score)
        gs = by_enc.get("__shared__", {}).get("graph")
        if gs is not None:
            by_cell[cell]["graph"].append(gs)
        for encoder in ENCODERS:
            for retr in ("dense", "rerank"):
                v = by_enc.get(encoder, {}).get(retr)
                if v is not None:
                    by_cell[cell][f"{retr}_{encoder}"].append(v)
    by_cell_means = {
        cell: {k: {"n": len(vs), "mean": mean(vs)} for k, vs in d.items() if vs}
        for cell, d in by_cell.items()
    }

    result = {
        "pooled": pooled,
        "by_cell": by_cell_means,
        "n_queries_with_scores": len(per_query),
        "n_cell_c_total": len(cell_c_meta),
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")

    # === Pretty print ===
    print("\n=== Pooled (Cell C rewritten vs original) ===")
    print(f"{'retriever':<24s} {'n':>5s} {'rewritten':>10s} {'original':>10s} {'Δ':>8s}")
    print("-" * 60)
    for k, v in pooled.items():
        sign = "+" if v["delta"] >= 0 else ""
        print(f"{k:<24s} {v['n']:>5d} {v['rewritten']:>10.4f} "
              f"{v['original']:>10.4f} {sign}{v['delta']:>7.4f}")

    print("\n=== By cell-of-origin (rewritten only) ===")
    for cell, d in by_cell_means.items():
        n_sample = list(d.values())[0]["n"] if d else 0
        print(f"  Cell {cell} (n≈{n_sample}):")
        for k, v in d.items():
            print(f"    {k:<24s} {v['mean']:.4f}")

    print("\n=== F8 pre-registered prediction check ===")
    pred = {
        "bm25": (-0.20, "−0.20"),
        "graph": (-0.45, "−0.45"),
        "dense_jina-1.5b": (-0.075, "−0.05 to −0.10"),
    }
    for k, (target, label) in pred.items():
        actual = pooled.get(k, {}).get("delta")
        if actual is None:
            continue
        match = "✓" if (actual <= target + 0.05 and actual >= target - 0.10) else "?"
        sign = "+" if actual >= 0 else ""
        print(f"  {k:<24s} predicted {label:>16s}  actual {sign}{actual:.4f}  {match}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
