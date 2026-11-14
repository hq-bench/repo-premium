#!/usr/bin/env python3
"""Aggregate the two new R8' 2022 experiments.

1) Qwen3-Embedding-4B dense (second-encoder falsification).
2) jina-0.5b + coreb-code-reranker overlay (RG1 stress test).

Reports pooled Δ vs jina-0.5b baseline; also breaks down rerank
gain by τ quartile to answer: does rerank rescue low-τ queries?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_ROOT
import json
from pathlib import Path
import numpy as np

BASE = (DATA_ROOT / 'smoke_test_results')

RUNS = {
    "jina-0.5b (baseline)": BASE / "r8_2022_jina-0.5b_c1000",
    "qwen3-4b (2nd encoder)": BASE / "r8_2022_qwen3-4b_c1000",
    "jina-0.5b + coreb-reranker": BASE / "r8_2022_rerank_jina-0.5b_c1000",
}

def load_run(run_dir: Path):
    per_q = []
    for repo_dir in sorted(run_dir.glob("*/")):
        f = repo_dir / "smoke_test_django.jsonl"
        if not f.exists():
            continue
        for line in f.open():
            r = json.loads(line)
            r["_repo"] = repo_dir.name
            per_q.append(r)
    return per_q

RNG = np.random.default_rng(42)
def bci(x, nb=5000, a=0.05):
    if len(x) < 2:
        return float("nan"), float("nan")
    means = np.array([RNG.choice(x, size=len(x), replace=True).mean() for _ in range(nb)])
    return float(np.quantile(means, a/2)), float(np.quantile(means, 1-a/2))

print("=" * 78)
print("R8' 2022 — three-condition aggregation")
print("=" * 78)

for name, path in RUNS.items():
    rows = load_run(path)
    if not rows:
        print(f"\n[{name}] NO DATA at {path}")
        continue
    print(f"\n[{name}]  path: {path}")
    print(f"  n_repos: {len(set(r['_repo'] for r in rows))}, n_queries: {len(rows)}")
    dense = np.array([r["dense_ndcg10"] for r in rows])
    bm25 = np.array([r["bm25_ndcg10"] for r in rows])
    delta = np.array([r["delta_ndcg10"] for r in rows])
    lo, hi = bci(delta)
    print(f"  dense nDCG@10 = {dense.mean():.4f}")
    print(f"  bm25  nDCG@10 = {bm25.mean():.4f}")
    print(f"  Δ (dense-bm25) = {delta.mean():+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")
    # leave-sympy-out
    ns_rows = [r for r in rows if "sympy" not in r["_repo"]]
    if len(ns_rows) < len(rows):
        ns_delta = np.array([r["delta_ndcg10"] for r in ns_rows])
        nlo, nhi = bci(ns_delta)
        print(f"  leave-sympy-out: n={len(ns_rows)}, Δ={ns_delta.mean():+.4f}, "
              f"95% CI [{nlo:+.4f}, {nhi:+.4f}]")
    # Rerank-specific
    if "rerank" in name.lower():
        rer = np.array([r["rerank_ndcg10"] for r in rows if r.get("rerank_ndcg10", -1) >= 0])
        if len(rer):
            drr = np.array([r["delta_rerank_ndcg10"] for r in rows if r.get("rerank_ndcg10", -1) >= 0])
            rr_lo, rr_hi = bci(drr)
            print(f"  rerank nDCG@10 = {rer.mean():.4f} (n={len(rer)})")
            print(f"  Δ_rerank (rerank - dense) = {drr.mean():+.4f}, "
                  f"95% CI [{rr_lo:+.4f}, {rr_hi:+.4f}]")
            # τ quartile breakdown of Δ_rerank
            taus = np.array([r["tau_q"] for r in rows if r.get("rerank_ndcg10", -1) >= 0])
            edges = np.quantile(taus, [0, 0.25, 0.5, 0.75, 1.0])
            print(f"  τ quartile edges: {[f'{e:.2f}' for e in edges]}")
            for i in range(4):
                mask = (taus >= edges[i]) & (taus <= edges[i+1] if i == 3 else taus < edges[i+1])
                if not mask.any():
                    continue
                sub = drr[mask]
                slo, shi = bci(sub)
                sub_dense = np.array([r["dense_ndcg10"] for r in rows if r.get("rerank_ndcg10", -1) >= 0])[mask]
                sub_rer = np.array([r["rerank_ndcg10"] for r in rows if r.get("rerank_ndcg10", -1) >= 0])[mask]
                print(f"    Q{i+1} τ∈[{edges[i]:.2f},{edges[i+1]:.2f}]: n={int(mask.sum()):3d}  "
                      f"dense={sub_dense.mean():.3f}  rerank={sub_rer.mean():.3f}  "
                      f"Δ_rr={sub.mean():+.4f}  CI [{slo:+.4f},{shi:+.4f}]")

print()
print("=" * 78)
print("Cross-encoder sanity: same-query paired comparison")
print("=" * 78)
rr_run = load_run(RUNS["jina-0.5b + coreb-reranker"])
qids_rr = {r["query_id"]: r for r in rr_run if r.get("rerank_ndcg10", -1) >= 0}
paired = [(q["dense_ndcg10"], q["rerank_ndcg10"]) for q in qids_rr.values()]
paired = np.array(paired)
n = len(paired)
if n:
    d = paired[:, 1] - paired[:, 0]
    lo, hi = bci(d)
    print(f"n paired queries: {n}")
    print(f"mean(rerank − dense) = {d.mean():+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"share where rerank ≥ dense: {(d >= 0).mean()*100:.1f}%")
    print(f"share where rerank > dense: {(d > 0).mean()*100:.1f}%")
    print(f"share where rerank < dense: {(d < 0).mean()*100:.1f}%")
