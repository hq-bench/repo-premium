"""Lockbox sensitivity (post hoc, not preregistered): how much of the fusion results rests on the
three single-repository R9 slices that hold 80% of the queries?

Uses gen_tier_lockbox.rows() unchanged (source-file labels when run on the indexed mirror) and
reports, for rrf_db - dense, rrf_dg - dense and rrf_dg - rrf_db:
  query-weighted   the preregistered estimand (pooled mean, repository-clustered bootstrap)
  repo-macro       mean of per-repository means, every repository weighted equally (bootstrap over repositories)
  pool-macro       mean of the four per-pool means (DeepSWE, TypeScript, Go, Rust)
  leave-one-out    the query-weighted estimate with each large repository dropped, and with all three dropped
Run from a root whose ${REPO_PREMIUM_DATA:-data}/ holds the lockbox caches. Writes analysis/lockbox_sensitivity.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_tier_lockbox as G  # noqa: E402

CONTRASTS = {"rrf_db_minus_dense": ("rrf_db", "dense"), "rrf_dg_minus_dense": ("rrf_dg", "dense"),
             "rrf_dg_minus_rrf_db": ("rrf_dg", "rrf_db")}
N, SEED = 10000, 0


def repo_means(rs, a, b):
    by = defaultdict(list)
    for r in rs:
        by[r["repo"]].append(r[a] - r[b])
    return {k: float(np.mean(v)) for k, v in by.items()}


def query_weighted(rs, a, b):
    v = np.array([r[a] - r[b] for r in rs])
    ci, _ = G.boot(rs, v, seed=SEED, n=N)
    return {"mean": float(v.mean()), "ci95": ci, "n": len(rs), "n_repos": len({r["repo"] for r in rs})}


def repo_macro(rs, a, b):
    m = np.array(list(repo_means(rs, a, b).values()))
    rng = np.random.default_rng(SEED)
    bs = np.array([m[rng.integers(0, len(m), len(m))].mean() for _ in range(N)])
    return {"mean": float(m.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n_repos": len(m)}


def main() -> int:
    rs, guard = G.rows()
    pools = sorted({r["pool"] for r in rs})
    big = sorted({r["repo"] for r in rs if r["pool"] != "deepswe"})
    out = {"guard": guard, "pools": {p: sum(r["pool"] == p for r in rs) for p in pools}, "contrasts": {}}
    for name, (a, b) in CONTRASTS.items():
        c = {"query_weighted": query_weighted(rs, a, b), "repo_macro": repo_macro(rs, a, b),
             "pool_macro": float(np.mean([np.mean([r[a] - r[b] for r in rs if r["pool"] == p]) for p in pools])),
             "leave_one_out": {}}
        for rp in big:
            c["leave_one_out"][rp] = query_weighted([r for r in rs if r["repo"] != rp], a, b)
        c["leave_one_out"]["all_three"] = query_weighted([r for r in rs if r["repo"] not in big], a, b)
        out["contrasts"][name] = c
        q, m = c["query_weighted"], c["repo_macro"]
        print(f"== {name}: query-weighted {q['mean']:+.4f} [{q['ci95'][0]:+.4f},{q['ci95'][1]:+.4f}] "
              f"| repo-macro {m['mean']:+.4f} [{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}] (n_repos={m['n_repos']}) "
              f"| pool-macro {c['pool_macro']:+.4f}")
        for k, v in c["leave_one_out"].items():
            print(f"     drop {k:28s} {v['mean']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] n={v['n']}")
    p = G.D / "analysis/lockbox_sensitivity.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    print("guard", guard, "pools", out["pools"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
