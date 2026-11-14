#!/usr/bin/env python3
"""Generative arm re-baselined against BM25 at its own best chunk size.

chunk_rebase_ext.py showed that c=1000 handicaps BM25 (best: whole files), so
the genarm closure figures computed at c=1000 measure generation against a
weak lexical baseline. Here both lexical arms run on whole files:

    premium_tuned  = dense(c=1000) - BM25_whole(original query)
    residual_tuned = dense(c=1000) - BM25_whole(query + generated anchors)
    closed_share   = (BM25_whole(gen) - BM25_whole(orig)) / premium_tuned

Dense is the canonical encoder per cell (as in tost_claims.genarm_claims).
BM25 runs reuse the genarm_bm25.slurm invocation (control: problem_statement
with --strip-boilerplate; expanded: expanded_query) at c = 1e9; the c=1000
runs of the same script must reproduce the cached genarm BM25 exactly.

CIs: repository-cluster bootstrap (10,000, seed 20260813, the genarm
procedure); equivalence at +/-0.05 on the 90% interval; share CI is a
ratio-of-sums repository bootstrap.
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

D = DATA_ROOT
SMOKE = D / "smoke_test_results"
CANON = SMOKE / "canon202607"
PULL = D / "remote_pull/genarm/results"
GEN = SMOKE / "gen_sweep202610"
SWEEP = SMOKE / "bm25_sweep202610"
WHOLE = 1000000000
DENSE = {"A": CANON / "jina05_cellA_c1000", "B": CANON / "jina05_cellB_c1000",
         "RA": SMOKE / "r8_robust_jina-0.5b_c1000", "RB": SMOKE / "r8_2022_jina-0.5b_c1000"}
DELTA = 0.05
SEED = 20260813


def load(d: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for f in sorted(d.glob("*/smoke_test_django.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[(f.parent.name, r["query_id"])] = r
    return out


def control_whole(cell: str) -> dict:
    # A/B: the canonical whole-file sweep (the genarm control arm reproduces the
    # canonical BM25 exactly); RA/RB: the control arm rerun on whole files.
    if cell in ("A", "B"):
        return load(SWEEP / f"bm25_cell{cell}_c{WHOLE}")
    return load(GEN / f"control_cell{cell}_c{WHOLE}")


def gen_dir(cell: str, c: int) -> Path:
    return GEN / f"gen_cell{cell}_c{c}"


def draws(clusters: list[list[float]], n_boot: int) -> list[float]:
    rng = random.Random(SEED)
    out = []
    for _ in range(n_boot):
        s: list[float] = []
        for _ in clusters:
            s.extend(clusters[rng.randrange(len(clusters))])
        out.append(sum(s) / len(s))
    out.sort()
    return out


def q(d: list[float], p: float) -> float:
    return d[min(len(d) - 1, int(p * len(d)))]


def summary(by_repo: dict[str, list[float]], n_boot: int, tost: bool = False) -> dict:
    vals = [v for g in by_repo.values() for v in g]
    d = draws(list(by_repo.values()), n_boot)
    out = {"mean": sum(vals) / len(vals), "ci95": [q(d, .025), q(d, .975)],
           "ci90": [q(d, .05), q(d, .95)]}
    if tost:
        out["tost_p"] = max(sum(x >= DELTA for x in d), sum(x <= -DELTA for x in d)) / len(d)
        out["equivalent"] = -DELTA < out["ci90"][0] and out["ci90"][1] < DELTA
    return out


def share(num: dict[str, list[float]], den: dict[str, list[float]], n_boot: int) -> dict:
    repos = sorted(num)
    rng = random.Random(SEED)
    st = []
    for _ in range(n_boot):
        a = b = 0.0
        for _ in repos:
            r = repos[rng.randrange(len(repos))]
            a += sum(num[r])
            b += sum(den[r])
        if b > 0:
            st.append(a / b)
    st.sort()
    tot = sum(map(sum, num.values())) / sum(map(sum, den.values()))
    return {"share": tot, "ci95": [q(st, .025), q(st, .975)], "valid_draws": len(st)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", type=Path, default=D / "analysis/genarm_rebase.json")
    args = ap.parse_args()
    res: dict = {"guards": {}, "cells": {}}
    for cell in ("A", "B", "RA", "RB"):
        dense = load(DENSE[cell])
        ctrl_c = load(PULL / f"control_cell{cell}")
        gen_c = load(PULL / f"expanded_cell{cell}")
        gen_c_new = load(gen_dir(cell, 1000))
        g = {"gen_c1000_max_abs_diff": max(abs(gen_c_new[k]["bm25_ndcg10"] - gen_c[k]["bm25_ndcg10"])
                                           for k in gen_c), "n": len(gen_c), "n_new": len(gen_c_new)}
        if cell in ("RA", "RB"):
            ctrl_c_new = load(GEN / f"control_cell{cell}_c1000")
            off = [k for k in ctrl_c if abs(ctrl_c_new[k]["bm25_ndcg10"] - ctrl_c[k]["bm25_ndcg10"]) > 1e-12]
            # BM25 score ties can be ordered differently by numpy on macOS vs the
            # cluster; accept only pure reorderings of an identical top-10 set.
            g["control_c1000_tie_reorders"] = [k[1] for k in off
                                               if set(ctrl_c_new[k]["bm25_top10"]) == set(ctrl_c[k]["bm25_top10"])]
            g["control_c1000_real_diffs"] = len(off) - len(g["control_c1000_tie_reorders"])
        res["guards"][cell] = g
        if g["n_new"] != g["n"] or g["gen_c1000_max_abs_diff"] > 1e-12 or g.get("control_c1000_real_diffs", 0):
            raise SystemExit(f"guard failed for {cell}: {g}")

        cw, gw = control_whole(cell), load(gen_dir(cell, WHOLE))
        keys = sorted(set(dense) & set(ctrl_c) & set(gen_c) & set(cw) & set(gw))
        R = defaultdict(lambda: defaultdict(list))
        for k in keys:
            dn = dense[k]["dense_ndcg10"]
            b0, b1 = cw[k]["bm25_ndcg10"], gw[k]["bm25_ndcg10"]
            c0, c1 = ctrl_c[k]["bm25_ndcg10"], gen_c[k]["bm25_ndcg10"]
            for name, v in (("premium_tuned", dn - b0), ("residual_tuned", dn - b1),
                            ("gain_tuned", b1 - b0), ("premium_c1000", dn - c0),
                            ("residual_c1000", dn - c1)):
                R[name][k[0]].append(v)
        cellres = {"n": len(keys), "n_repos": len({k[0] for k in keys}),
                   "means": {"dense": sum(dense[k]["dense_ndcg10"] for k in keys) / len(keys),
                             "bm25_whole": sum(cw[k]["bm25_ndcg10"] for k in keys) / len(keys),
                             "gen_whole": sum(gw[k]["bm25_ndcg10"] for k in keys) / len(keys)}}
        for name in R:
            cellres[name] = summary(R[name], args.n_boot, tost=name.startswith("residual"))
        cellres["closed_share_tuned"] = share(R["gain_tuned"], R["premium_tuned"], args.n_boot)
        res["cells"][cell] = cellres
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    print("guards:", json.dumps(res["guards"]))
    for cell, c in res["cells"].items():
        m = c["means"]
        f = lambda s: f"{s['mean']:+.3f} [{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]"
        print(f"{cell:2s} n={c['n']:3d} dense={m['dense']:.3f} bm25w={m['bm25_whole']:.3f} "
              f"genw={m['gen_whole']:.3f} | premium {f(c['premium_tuned'])} -> residual "
              f"{f(c['residual_tuned'])} (90% {[round(x, 3) for x in c['residual_tuned']['ci90']]}, "
              f"equiv={c['residual_tuned']['equivalent']}) | closed "
              f"{c['closed_share_tuned']['share']:.0%} "
              f"[{c['closed_share_tuned']['ci95'][0]:.0%},{c['closed_share_tuned']['ci95'][1]:.0%}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
