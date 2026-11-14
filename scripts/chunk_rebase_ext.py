#!/usr/bin/env python3
"""Extended BM25 chunk sweep (Cells A and B) and the premium against a tuned BM25.

chunk_rebase.py found BM25's best c on Cell A at the edge of its grid (3000)
and could not rebase Cell B at all (no sweep). This script reads a BM25-only
sweep run with the canonical invocation (canonical_sweep.slurm flags plus
--skip-dense):

    Cell A: c in {1000, 3000, 4000, 6000, 10000, whole file}
    Cell B: c in {1000, 1500, 2000, 3000, 4000, 6000, 10000, whole file}

Guards (must be exact, per query): BM25 at A/c1000, A/c3000 and B/c1000
reproduces the cached canon202607 jina05 runs.

Reported per cell:
  * the BM25 curve over c;
  * every encoder's premium (dense at c=1000) vs BM25 at c=1000 and vs BM25 at
    its best c (repo-block and per-query bootstrap, chunk_rebase.py procedure).
    Picking BM25's best c on the evaluation queries favours BM25, so the
    rebased premium is conservative;
  * for the generative arm, the share of the premium it closes,
    (bm25_expanded - bm25_control) / (dense - bm25_control), all at c=1000
    (the configuration the arm was run in), with a repo-block ratio CI.
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import glob
import json
import os
import random
from pathlib import Path

D = DATA_ROOT
CANON = D / "smoke_test_results/canon202607"
SMOKE = D / "smoke_test_results"
PULL = D / "remote_pull"
NOFN = D / "genarm_nofn/results"
WHOLE = 1000000000
GRID = {"A": (1000, 3000, 4000, 6000, 10000, WHOLE),
        "B": (1000, 1500, 2000, 3000, 4000, 6000, 10000, WHOLE)}
GENARM_DENSE = {"A": CANON / "jina05_cellA_c1000", "B": CANON / "jina05_cellB_c1000",
                "RA": SMOKE / "r8_robust_jina-0.5b_c1000", "RB": SMOKE / "r8_2022_jina-0.5b_c1000"}


def load(d: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    for f in sorted(glob.glob(str(d / "*" / "smoke_test_django.jsonl"))):
        repo = os.path.basename(os.path.dirname(f))
        for line in open(f):
            r = json.loads(line)
            rows[(repo, r["query_id"])] = r
    return rows


def mean(xs):
    return sum(xs) / len(xs)


def by_repo(keys):
    g: dict[str, list] = {}
    for k in keys:
        g.setdefault(k[0], []).append(k)
    return g


def boot_mean(diff: dict, n_boot: int, seed: int, clustered: bool) -> dict:
    """chunk_rebase.boot, unchanged."""
    keys = sorted(diff)
    rng = random.Random(seed)
    stats = []
    if clustered:
        g = {r: [diff[k] for k in ks] for r, ks in by_repo(keys).items()}
        repos = sorted(g)
        for _ in range(n_boot):
            xs = []
            for _ in repos:
                xs.extend(g[rng.choice(repos)])
            stats.append(mean(xs))
    else:
        vals = [diff[k] for k in keys]
        n = len(vals)
        for _ in range(n_boot):
            stats.append(mean([vals[rng.randrange(n)] for _ in range(n)]))
    stats.sort()

    def q(p):
        return stats[min(len(stats) - 1, max(0, int(round(p * (len(stats) - 1)))))]

    return {"ci95": [q(0.025), q(0.975)], "ci90": [q(0.05), q(0.95)]}


def premium(dense_rows, bm25_rows, n_boot, seed):
    keys = sorted(set(dense_rows) & set(bm25_rows))
    diff = {k: dense_rows[k]["dense_ndcg10"] - bm25_rows[k]["bm25_ndcg10"] for k in keys}
    return {"n": len(keys), "n_repos": len(by_repo(keys)),
            "premium": mean(list(diff.values())),
            "repo_boot": boot_mean(diff, n_boot, seed, True),
            "query_boot": boot_mean(diff, n_boot, seed, False)}


def closed_share(num: dict, den: dict, n_boot: int, seed: int) -> dict:
    """Ratio of means sum(num)/sum(den), repository-block bootstrap."""
    keys = sorted(num)
    g = by_repo(keys)
    repos = sorted(g)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        n = d = 0.0
        for _ in repos:
            for k in g[rng.choice(repos)]:
                n += num[k]
                d += den[k]
        stats.append(n / d if d > 0 else float("nan"))
    stats = sorted(s for s in stats if s == s)

    def q(p):
        return stats[min(len(stats) - 1, max(0, int(round(p * (len(stats) - 1)))))]

    return {"share": sum(num.values()) / sum(den.values()),
            "ci95": [q(0.025), q(0.975)], "ci90": [q(0.05), q(0.95)],
            "n": len(keys), "n_valid_draws": len(stats)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-root", type=Path, default=SMOKE / "bm25_sweep202610",
                    help="dir holding bm25_cell{A,B}_c{C}/<repo>/smoke_test_django.jsonl")
    ap.add_argument("--out", type=Path, default=D / "analysis/chunk_rebase_ext.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()

    out: dict = {"grid": GRID, "guards": {}, "sweep": {}, "rebased": {}, "genarm_closed": {}}

    sweep = {cell: {c: load(args.sweep_root / f"bm25_cell{cell}_c{c}") for c in grid}
             for cell, grid in GRID.items()}

    for cell, c in (("A", 1000), ("A", 3000), ("B", 1000)):
        ref = load(CANON / f"jina05_cell{cell}_c{c}")
        new = sweep[cell][c]
        diffs = [abs(new[k]["bm25_ndcg10"] - ref[k]["bm25_ndcg10"]) for k in ref if k in new]
        out["guards"][f"{cell}_c{c}"] = {"n_ref": len(ref), "n_new": len(new),
                                         "n_common": len(diffs),
                                         "max_abs_diff": max(diffs) if diffs else None}
    for g in out["guards"].values():
        if g["n_common"] != g["n_ref"] or g["n_new"] != g["n_ref"] or g["max_abs_diff"] > 1e-12:
            raise SystemExit(f"guard failed: {out['guards']}")

    for cell, grid in GRID.items():
        curve = {c: mean([r["bm25_ndcg10"] for r in sweep[cell][c].values()]) for c in grid}
        c_best = max(curve, key=lambda c: curve[c])
        out["sweep"][cell] = {"bm25": curve, "n": {c: len(sweep[cell][c]) for c in grid},
                              "c_best": c_best, "c_best_on_grid_edge": c_best == grid[-1]}
        enc = {}
        for p in sorted(CANON.iterdir()):
            if not p.name.endswith(f"_cell{cell}_c1000"):
                continue
            rows = load(p)
            enc[p.name.split("_cell")[0]] = {
                "vs_bm25_c1000": premium(rows, sweep[cell][1000], args.n_boot, args.seed),
                "vs_bm25_best": premium(rows, sweep[cell][c_best], args.n_boot, args.seed),
            }
        out["rebased"][cell] = enc

    for cell, ddir in GENARM_DENSE.items():
        dense = load(ddir)
        ctrl = load(PULL / "genarm/results" / f"control_cell{cell}")
        arms = {"expanded": load(PULL / "genarm/results" / f"expanded_cell{cell}"),
                "nofn": load(NOFN / f"nofn_cell{cell}")}
        res = {}
        for arm, rows in arms.items():
            keys = sorted(set(dense) & set(ctrl) & set(rows))
            num = {k: rows[k]["bm25_ndcg10"] - ctrl[k]["bm25_ndcg10"] for k in keys}
            den = {k: dense[k]["dense_ndcg10"] - ctrl[k]["bm25_ndcg10"] for k in keys}
            res[arm] = closed_share(num, den, args.n_boot, args.seed)
        out["genarm_closed"][cell] = res

    args.out.write_text(json.dumps(out, indent=2) + "\n")

    print("guards:", json.dumps(out["guards"]))
    for cell in GRID:
        s = out["sweep"][cell]
        print(f"\nCell {cell} BM25 curve:",
              {("whole" if c == WHOLE else c): round(v, 4) for c, v in s["bm25"].items()},
              f"best={s['c_best']} edge={s['c_best_on_grid_edge']}")
        for e, d in sorted(out["rebased"][cell].items(),
                           key=lambda kv: -kv[1]["vs_bm25_best"]["premium"]):
            a, b = d["vs_bm25_c1000"], d["vs_bm25_best"]
            print(f"  {e:9s} n={b['n']:3d} c1000 {a['premium']:+.3f} "
                  f"{[round(x, 3) for x in a['repo_boot']['ci95']]}  best "
                  f"{b['premium']:+.3f} {[round(x, 3) for x in b['repo_boot']['ci95']]}")
    print("\ngenarm share of premium closed (c=1000):")
    for cell, res in out["genarm_closed"].items():
        for arm, r in res.items():
            print(f"  {cell:2s} {arm:8s} {r['share']:.3f} 95={[round(x, 3) for x in r['ci95']]}"
                  f" valid={r['n_valid_draws']}")


if __name__ == "__main__":
    main()
