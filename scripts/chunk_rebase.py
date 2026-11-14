#!/usr/bin/env python3
"""Re-baseline the headline dense premium against BM25 at its OWN best chunk size.

Review #20: c=1000 was chosen partly because it maximises Delta, while BM25
improves monotonically with chunk size and dense stays flat. Here each
retriever is taken at its own best configuration on the swept grid:

    premium_rebased(q) = dense_enc(q; c_dense*) - bm25(q; c_bm25*)

BM25 does not depend on the encoder, so BM25 at c_bm25* is read from any cache
swept at that chunk size (only jina05 Cell A was swept). Dense for encoders
without a sweep is taken at c=1000; for jina05 the sweep shows dense is flat,
and its own best c is also reported.

CIs follow the paper's per-query procedure (paired bootstrap over queries,
10,000 resamples, fixed seed); a repository-clustered paired bootstrap is
reported alongside, since queries in one repository share a corpus and IDF.

Loader check: the c=1000 premiums must reproduce tab:cells / tab:r5
(+0.121 Cell A, +0.204 Cell B; BM25 0.411/0.427/0.443/0.465 over the sweep).
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import glob
import json
import os
import random
from pathlib import Path

GRID = (1000, 1500, 2000, 3000)


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


def boot(diff: dict[tuple[str, str], float], n_boot: int, seed: int, clustered: bool):
    keys = sorted(diff)
    rng = random.Random(seed)
    stats = []
    if clustered:
        by_repo: dict[str, list[float]] = {}
        for k in keys:
            by_repo.setdefault(k[0], []).append(diff[k])
        repos = sorted(by_repo)
        for _ in range(n_boot):
            xs = []
            for _ in repos:
                xs.extend(by_repo[rng.choice(repos)])
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
    return {
        "n": len(keys),
        "n_dense_only": len(set(dense_rows) - set(bm25_rows)),
        "n_bm25_only": len(set(bm25_rows) - set(dense_rows)),
        "dense": mean([dense_rows[k]["dense_ndcg10"] for k in keys]),
        "bm25": mean([bm25_rows[k]["bm25_ndcg10"] for k in keys]),
        "premium": mean(list(diff.values())),
        "query_boot": boot(diff, n_boot, seed, clustered=False),
        "repo_boot": boot(diff, n_boot, seed, clustered=True),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path,
                    default=(DATA_ROOT / 'smoke_test_results/canon202607'))
    ap.add_argument("--out", type=Path,
                    default=(DATA_ROOT / 'analysis/chunk_rebase.json'))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()

    caches = {p.name: load(p) for p in sorted(args.root.iterdir()) if p.is_dir()}
    out: dict = {"grid": GRID, "sweep": {}, "check": {}, "rebased": {}, "missing": []}

    # Loader check against tab:cells.
    for cell, want in (("A", 0.121), ("B", 0.204)):
        r = caches[f"jina05_cell{cell}_c1000"]
        p = mean([x["dense_ndcg10"] - x["bm25_ndcg10"] for x in r.values()])
        out["check"][cell] = {"premium_c1000": p, "published": want,
                              "ok": abs(round(p, 3) - want) < 1.5e-3}

    for cell in ("A", "B"):
        sweep = {}
        for c in GRID:
            name = f"jina05_cell{cell}_c{c}"
            if name not in caches:
                out["missing"].append(f"BM25 (and jina05 dense) Cell {cell} c={c}")
                continue
            r = caches[name]
            sweep[c] = {"n": len(r),
                        "bm25": mean([x["bm25_ndcg10"] for x in r.values()]),
                        "dense": mean([x["dense_ndcg10"] for x in r.values()])}
        out["sweep"][cell] = sweep
        if len(sweep) < 2:
            out["rebased"][cell] = None
            continue
        c_bm = max(sweep, key=lambda c: sweep[c]["bm25"])
        c_de = max(sweep, key=lambda c: sweep[c]["dense"])
        bm_rows = caches[f"jina05_cell{cell}_c{c_bm}"]
        res = {"c_bm25_best": c_bm, "c_bm25_best_on_grid_edge": c_bm == GRID[-1],
               "c_jina05_dense_best": c_de, "encoders": {}}
        for name, rows in caches.items():
            if not name.endswith(f"_cell{cell}_c1000"):
                continue
            enc = name.split("_cell")[0]
            res["encoders"][enc] = {
                "c1000_vs_c1000": premium(rows, rows, args.n_boot, args.seed),
                f"c1000_vs_bm25_c{c_bm}": premium(rows, bm_rows, args.n_boot, args.seed),
            }
        res["encoders"]["jina05"][f"c{c_de}_vs_bm25_c{c_bm}"] = premium(
            caches[f"jina05_cell{cell}_c{c_de}"], bm_rows, args.n_boot, args.seed)
        out["rebased"][cell] = res

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    for cell, chk in out["check"].items():
        print(f"check Cell {cell}: {chk['premium_c1000']:+.4f} vs published "
              f"{chk['published']:+.3f} -> {'OK' if chk['ok'] else 'MISMATCH'}")
    for cell, res in out["rebased"].items():
        print(f"\nCell {cell} sweep:", {c: round(v['bm25'], 4) for c, v in out['sweep'][cell].items()})
        if res is None:
            print(f"  Cell {cell}: no chunk sweep -> cannot rebase")
            continue
        print(f"  BM25 best c={res['c_bm25_best']} (grid edge: {res['c_bm25_best_on_grid_edge']})")
        for enc, d in sorted(res["encoders"].items()):
            for k, v in d.items():
                print(f"  {enc:9s} {k:22s} n={v['n']} prem={v['premium']:+.4f} "
                      f"q95={[round(x, 4) for x in v['query_boot']['ci95']]} "
                      f"repo95={[round(x, 4) for x in v['repo_boot']['ci95']]}")
    print("\nmissing:", out["missing"])


if __name__ == "__main__":
    main()
