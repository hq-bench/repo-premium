"""R8 negative-control aggregation (mined pre-cutoff PRs, 2023).

Reads ${REPO_PREMIUM_DATA:-data}/coreb_repo_r5_results/r5_jina-0.5b_c1000/<repo>/smoke_test_django.jsonl
(one row per query with grep_/bm25_/dense_ndcg10 fields), reports per-repo +
pooled mean nDCG@10 and paired Δ = dense − BM25.

Bootstrap CI on pooled Δ at 95 %.

Compares against canonical chunk=1000 stripped pipeline:
  Cell A stripped Δ = +0.121 (curated, pre-cutoff)      ← F2 at c=1000
  Cell B stripped Δ = +0.206 (mined,   post-cutoff)     ← F5 at c=1000

If R8 (pre-cutoff mined) ≈ Cell A → field gap is a *date* effect → F2 survives.
If R8 ≈ Cell B stripped → field gap is a *mining-methodology* effect → F2 weakens.
If R8 midway → both effects contribute additively.

Output: console table + JSON summary.

Usage:
  python scripts/aggregate_r5.py \\
      --r5-root ${REPO_PREMIUM_DATA:-data}/coreb_repo_r5_results \\
      --encoder jina-0.5b \\
      --out ${REPO_PREMIUM_DATA:-data}/coreb_repo_r5_aggregated.json
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
import statistics
from pathlib import Path
from random import Random


REPOS = [
    "django__django", "sympy__sympy", "sphinx-doc__sphinx", "matplotlib__matplotlib",
    "scikit-learn__scikit-learn", "astropy__astropy", "pydata__xarray",
    "pytest-dev__pytest", "pylint-dev__pylint", "psf__requests",
]


def load_repo(root: Path, encoder: str, repo: str) -> list[dict]:
    p = root / f"r5_{encoder}_c1000" / repo / "smoke_test_django.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def bootstrap_ci(deltas: list[float], n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    rng = Random(seed)
    means = []
    n = len(deltas)
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return sum(deltas) / n, lo, hi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r5-root", type=Path, default=(DATA_ROOT / 'smoke_test_results'))
    ap.add_argument("--encoder", default="jina-0.5b")
    ap.add_argument("--out", type=Path, default=(DATA_ROOT / 'coreb_repo_r5_aggregated.json'))
    args = ap.parse_args()

    per_repo = {}
    pooled_grep, pooled_bm25, pooled_dense, pooled_delta = [], [], [], []

    print(f"{'repo':<32s} {'n':>4s} {'grep':>7s} {'bm25':>7s} {'dense':>7s} {'Δ':>7s}")
    print("-" * 75)

    for repo in REPOS:
        rows = load_repo(args.r5_root, args.encoder, repo)
        if not rows:
            print(f"{repo:<32s} (no rows)")
            per_repo[repo] = {"n": 0}
            continue

        grep = [r["grep_ndcg10"] for r in rows]
        bm25 = [r["bm25_ndcg10"] for r in rows]
        dense = [r["dense_ndcg10"] for r in rows]
        delta = [d - b for d, b in zip(dense, bm25)]
        n = len(rows)
        m_grep = sum(grep) / n
        m_bm25 = sum(bm25) / n
        m_dense = sum(dense) / n
        m_delta = sum(delta) / n
        per_repo[repo] = {
            "n": n,
            "grep": m_grep, "bm25": m_bm25, "dense": m_dense, "delta": m_delta,
        }
        pooled_grep.extend(grep)
        pooled_bm25.extend(bm25)
        pooled_dense.extend(dense)
        pooled_delta.extend(delta)
        print(f"{repo:<32s} {n:>4d} {m_grep:>7.3f} {m_bm25:>7.3f} {m_dense:>7.3f} {m_delta:>+7.3f}")

    n_pool = len(pooled_delta)
    if n_pool == 0:
        print("\nNo data collected.")
        return 1

    mean_delta, ci_lo, ci_hi = bootstrap_ci(pooled_delta)
    pool_grep = sum(pooled_grep) / n_pool
    pool_bm25 = sum(pooled_bm25) / n_pool
    pool_dense = sum(pooled_dense) / n_pool
    sd = statistics.stdev(pooled_delta) if n_pool > 1 else 0.0
    se = sd / (n_pool ** 0.5) if n_pool > 1 else 0.0

    print("-" * 75)
    print(f"{'POOLED':<32s} {n_pool:>4d} {pool_grep:>7.3f} {pool_bm25:>7.3f} {pool_dense:>7.3f} {mean_delta:>+7.3f}")
    print()
    print(f"Pooled Δ = {mean_delta:+.3f}   95%CI(boot) = [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"             SD={sd:.3f}   SE={se:.4f}   n={n_pool}")
    print()
    CELL_A = 0.121   # canonical c=1000 stripped, jina-0.5b
    CELL_B = 0.206   # canonical c=1000 stripped, jina-0.5b
    print(f"  vs Cell A stripped c=1000 (pre-cutoff curated) +{CELL_A:.3f}  "
          f"→ Δ_R8 − Δ_A = {mean_delta - CELL_A:+.3f}")
    print(f"  vs Cell B stripped c=1000 (post-cutoff mined)  +{CELL_B:.3f}  "
          f"→ Δ_R8 − Δ_B = {mean_delta - CELL_B:+.3f}")
    print()
    d_to_a = abs(mean_delta - CELL_A)
    d_to_b = abs(mean_delta - CELL_B)
    in_b_band = ci_lo <= CELL_B <= ci_hi
    excludes_a = not (ci_lo <= CELL_A <= ci_hi)
    # Sympy-leave-out sensitivity check (heavy single-repo share is a known confound)
    sympy_rows = per_repo.get("sympy__sympy", {})
    if sympy_rows.get("n", 0) > 0:
        nosympy_delta = []
        for repo in REPOS:
            if repo == "sympy__sympy":
                continue
            rs = load_repo(args.r5_root, args.encoder, repo)
            for r in rs:
                nosympy_delta.append(r["dense_ndcg10"] - r["bm25_ndcg10"])
        ns_mean = sum(nosympy_delta) / len(nosympy_delta) if nosympy_delta else float("nan")
        print(f"Leave-sympy-out:  Δ = {ns_mean:+.3f}  (n={len(nosympy_delta)})  "
              f"vs A={CELL_A:+.3f}, B={CELL_B:+.3f}")
        summary_leave_sympy = {
            "n": len(nosympy_delta),
            "delta_mean": ns_mean,
            "delta_a_diff": ns_mean - CELL_A,
            "delta_b_diff": ns_mean - CELL_B,
        }
    else:
        summary_leave_sympy = None
    print()
    if d_to_b < d_to_a and in_b_band and excludes_a:
        verdict = ("R8 pooled matches CELL B stripped (95%CI excludes Cell A). "
                   "Field gap is dominantly a *mining-methodology* effect; date effect "
                   "is small. F2 needs major caveat: the post-cutoff signal is mostly "
                   "raw-PR-description-vs-curated-problem-statement, not training-data "
                   "leakage. Per-repo picture is mixed (driven heavily by sympy at "
                   f"{sympy_rows.get('n',0)}/{n_pool} of the pool, R8 sympy Δ = "
                   f"{sympy_rows.get('delta', float('nan')):+.3f} vs Cell A sympy "
                   "+0.064). Recommend: re-frame F2 as 'raw PR descriptions inflate Δ' "
                   "and report the small residual date effect separately if any.")
    elif d_to_a < d_to_b:
        verdict = ("R8 matches CELL A — Cell B's gap is a *date* effect. F2 SURVIVES.")
    else:
        verdict = ("R8 sits between Cell A and Cell B — both mining methodology and "
                   "post-cutoff date contribute additively. F2 needs caveat language: "
                   "decompose the field gap into mining + date components.")
    print(f"Verdict: {verdict}")

    summary = {
        "encoder": args.encoder,
        "n_pooled": n_pool,
        "pooled": {
            "grep": pool_grep, "bm25": pool_bm25, "dense": pool_dense,
            "delta_mean": mean_delta, "delta_sd": sd, "delta_se": se,
            "delta_ci95_lo": ci_lo, "delta_ci95_hi": ci_hi,
        },
        "per_repo": per_repo,
        "reference": {
            "cell_a_stripped_c1000_pooled_delta": 0.121,
            "cell_b_stripped_c1000_pooled_delta": 0.206,
            "note": "Apples-to-apples baseline: stripped + chunk=1000 + jina-code-0.5b",
        },
        "leave_sympy_out": summary_leave_sympy,
        "verdict": verdict,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
