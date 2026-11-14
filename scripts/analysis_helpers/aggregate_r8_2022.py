#!/usr/bin/env python3
"""Aggregate R8 2022 smoke_test results across 10 Cell A repos."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_ROOT
import json
from pathlib import Path
import numpy as np

BASE = (DATA_ROOT / 'smoke_test_results/r8_2022_jina-0.5b_c1000')

rows = []
tot_n = 0
wt_delta = 0.0
wt_dense = 0.0
wt_bm25 = 0.0
for p in sorted(BASE.glob("*/smoke_test_django_summary.json")):
    repo = p.parent.name
    s = json.loads(p.read_text())
    n = s["n_queries"]
    d = s["ndcg10"]["dense"]
    b = s["ndcg10"]["bm25"]
    delta = s["ndcg10"]["premium_dense_minus_bm25"]
    rows.append((repo, n, d, b, delta))
    tot_n += n
    wt_delta += n * delta
    wt_dense += n * d
    wt_bm25 += n * b

print(f"{'repo':30s}{'n':>5s}{'dense':>10s}{'bm25':>10s}{'delta':>10s}")
print("-" * 65)
for repo, n, d, b, delta in rows:
    print(f"{repo:30s}{n:>5d}{d:>10.4f}{b:>10.4f}{delta:>+10.4f}")
print("-" * 65)

if tot_n:
    print(f"{'POOLED':30s}{tot_n:>5d}{wt_dense/tot_n:>10.4f}{wt_bm25/tot_n:>10.4f}{wt_delta/tot_n:>+10.4f}")
    tot_ns = wds = wbs = wdel = 0
    for repo, n, d, b, delta in rows:
        if "sympy" in repo:
            continue
        tot_ns += n; wds += n * d; wbs += n * b; wdel += n * delta
    if tot_ns:
        label = "POOLED_no_sympy"
        print(f"{label:30s}{tot_ns:>5d}{wds/tot_ns:>10.4f}{wbs/tot_ns:>10.4f}{wdel/tot_ns:>+10.4f}")

# Bootstrap CI on per-query deltas
per_q = []
for p in sorted(BASE.glob("*/smoke_test_django.jsonl")):
    for line in p.open():
        r = json.loads(line)
        if "delta_ndcg10" in r:
            per_q.append(r["delta_ndcg10"])
per_q = np.array(per_q)
print(f"\nn_per_query = {len(per_q)}, mean_delta = {per_q.mean():+.4f}")
rng = np.random.default_rng(42)
boots = np.array([rng.choice(per_q, size=len(per_q), replace=True).mean() for _ in range(5000)])
lo, hi = np.quantile(boots, [0.025, 0.975])
print(f"95%% bootstrap CI: [{lo:+.4f}, {hi:+.4f}]")

# Per-query deltas excluding sympy
per_q_ns = []
for p in sorted(BASE.glob("*/smoke_test_django.jsonl")):
    if "sympy" in p.parent.name:
        continue
    for line in p.open():
        r = json.loads(line)
        if "delta_ndcg10" in r:
            per_q_ns.append(r["delta_ndcg10"])
per_q_ns = np.array(per_q_ns)
print(f"\nno_sympy: n={len(per_q_ns)}, mean_delta = {per_q_ns.mean():+.4f}")
boots_ns = np.array([rng.choice(per_q_ns, size=len(per_q_ns), replace=True).mean() for _ in range(5000)])
lo, hi = np.quantile(boots_ns, [0.025, 0.975])
print(f"95%% bootstrap CI (no sympy): [{lo:+.4f}, {hi:+.4f}]")
