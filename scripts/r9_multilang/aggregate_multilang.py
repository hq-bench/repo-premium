#!/usr/bin/env python3
"""Aggregate R9B/R9C/R9D multi-language τ-effect replication.

Reports per-language quartile means of Δ(τ̂) and pooled cross-language
Δ with 95% bootstrap CI. Confirms whether Q1→Q4 monotonicity of the
τ-effect (established on Python R8/R8') replicates on JS/TS, Go, Rust.
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
    "TS (microsoft/TypeScript)": BASE / "r9b_ts_c1000",
    "Go (gohugoio/hugo)":        BASE / "r9c_go_c1000",
    "Rust (tokio-rs/tokio)":     BASE / "r9d_rust_c1000",
}

RNG = np.random.default_rng(42)

def bci(x, nb=5000, a=0.05):
    if len(x) < 2:
        return float("nan"), float("nan")
    x = np.asarray(x)
    means = np.array([RNG.choice(x, size=len(x), replace=True).mean()
                      for _ in range(nb)])
    return float(np.quantile(means, a / 2)), float(np.quantile(means, 1 - a / 2))

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

def tau_quartiles(rows, tau_key="tau_q"):
    taus = np.array([r[tau_key] for r in rows])
    deltas = np.array([r["delta_ndcg10"] for r in rows])
    if len(taus) < 4:
        return None
    edges = np.quantile(taus, [0, 0.25, 0.5, 0.75, 1.0])
    out = []
    for i in range(4):
        if i == 3:
            mask = (taus >= edges[i]) & (taus <= edges[i + 1])
        else:
            mask = (taus >= edges[i]) & (taus < edges[i + 1])
        if not mask.any():
            out.append(None)
            continue
        sub = deltas[mask]
        lo, hi = bci(sub)
        out.append((int(mask.sum()), float(sub.mean()), lo, hi,
                    float(edges[i]), float(edges[i + 1])))
    return out

print("=" * 78)
print("R9 multi-language τ-effect replication")
print("=" * 78)

all_deltas = []
per_lang_summary = []

for name, path in RUNS.items():
    rows = load_run(path)
    if not rows:
        print(f"\n[{name}] NO DATA at {path}")
        continue
    print(f"\n[{name}]  path: {path}")
    print(f"  n_queries: {len(rows)}")
    delta = np.array([r["delta_ndcg10"] for r in rows])
    dense = np.array([r["dense_ndcg10"] for r in rows])
    bm25 = np.array([r["bm25_ndcg10"] for r in rows])
    lo, hi = bci(delta)
    print(f"  dense nDCG@10 = {dense.mean():.4f}   "
          f"bm25 nDCG@10 = {bm25.mean():.4f}")
    print(f"  Δ (dense-bm25) = {delta.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    qs = tau_quartiles(rows)
    if qs is None:
        print("  <insufficient rows for quartile split>")
        continue
    print("  τ̂ quartile breakdown of Δ:")
    means = []
    for i, q in enumerate(qs, 1):
        if q is None:
            print(f"    Q{i}: <empty>")
            means.append(None)
            continue
        n, m, ql, qh, e0, e1 = q
        print(f"    Q{i} τ̂∈[{e0:.2f},{e1:.2f}]  n={n:3d}  "
              f"Δ={m:+.4f}  CI [{ql:+.4f},{qh:+.4f}]")
        means.append(m)
    if means[0] is not None and means[-1] is not None:
        shift = means[-1] - means[0]
        monotone = all(
            means[i] is not None and means[i + 1] is not None
            and means[i + 1] >= means[i] - 0.02  # tolerate 2pt wobble
            for i in range(3)
        )
        tag = "MONOTONE" if monotone else "NON-MONOTONE"
        print(f"  Q4 − Q1 shift: {shift:+.4f}    [{tag}]")
        per_lang_summary.append((name, shift, monotone, len(rows)))
    all_deltas.append(delta)

print()
print("=" * 78)
print("Pooled cross-language Δ")
print("=" * 78)
if all_deltas:
    pooled = np.concatenate(all_deltas)
    lo, hi = bci(pooled)
    print(f"n_pooled: {len(pooled)}")
    print(f"pooled Δ = {pooled.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

print()
print("=" * 78)
print("τ-effect replication summary")
print("=" * 78)
for name, shift, monotone, n in per_lang_summary:
    tag = "OK" if monotone and shift > 0 else "FAIL"
    print(f"  [{tag:>4}]  {name:<40}  Q4−Q1 = {shift:+.4f}  (n={n})")

n_ok = sum(1 for _, s, m, _ in per_lang_summary if m and s > 0)
print(f"\n{n_ok}/{len(per_lang_summary)} non-Python languages replicate the τ-effect.")
print("Combined with R8/R8' (Python) and R9 (JS, vuejs/core), this covers "
      "up to five languages.")
