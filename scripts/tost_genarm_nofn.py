#!/usr/bin/env python3
"""TOST for the residual premium after file names are stripped from the
generative expansion (genarm_strip_filenames.py, review finding #15).

genarm_strip_filenames.json tests whether stripping changes the premium; this
script tests the claim the paper actually makes about the residual itself:
"after generate-then-grep the premium is equivalent to zero" (|theta| < 0.05).

Same procedure as tost_claims.genarm_claims: canonical dense encoder per cell,
repository-cluster bootstrap, seed 20260813, 10k draws, same query set as the
strip analysis (so the 95% CIs reproduce genarm_strip_filenames.json first).
Holm is reported (a) within the four nofn claims and (b) within the genarm
family extended by these four (7 + 4 = 11).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tost_claims import (CANON, D, PULL, SMOKE, Claim, draws_cluster, holm,
                         load_run, mean, q_cluster)

NOFN = D / "genarm_nofn/results"
CELLS = {
    "A": CANON / "jina05_cellA_c1000",
    "B": CANON / "jina05_cellB_c1000",
    "RA": SMOKE / "r8_robust_jina-0.5b_c1000",
    "RB": SMOKE / "r8_2022_jina-0.5b_c1000",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", type=Path, default=D / "analysis/tost_genarm_nofn.json")
    args = ap.parse_args()

    strip = json.loads((D / "analysis/genarm_strip_filenames.json").read_text())["cells"]
    claims = []
    for cell, ddir in CELLS.items():
        ctrl = load_run(PULL / "genarm/results" / f"control_cell{cell}")
        exp = load_run(PULL / "genarm/results" / f"expanded_cell{cell}")
        rerun = load_run(NOFN / f"expanded_cell{cell}")
        nofn = load_run(NOFN / f"nofn_cell{cell}")
        dense = load_run(ddir)
        common = sorted(set(ctrl) & set(exp) & set(dense) & set(nofn) & set(rerun))
        groups: dict[str, list[float]] = defaultdict(list)
        for q in common:
            groups[ctrl[q]["repo"]].append(dense[q]["dense_ndcg10"] - nofn[q]["bm25_ndcg10"])
        vals = [v for g in groups.values() for v in g]
        pub = strip[cell]["premium_nofn"]
        claims.append(Claim(
            f"genarmnofn_canon_{cell}",
            f"post-expansion premium, file names stripped, Cell {cell}, "
            f"vs {ddir.name.split('_c1000')[0]} (n={len(vals)})",
            None, "equivalence", mean(vals),
            draws_cluster(list(groups.values()), args.n_boot, 20260813), q_cluster,
            (pub["mean"], *pub["ci95"])))
    res = [c.evaluate() for c in claims]
    holm(res, key="holm_nofn")

    prior = [r for r in json.loads((D / "analysis/tost_claims.json").read_text())["claims"]
             if r["id"].startswith("genarm_")]
    ext = [dict(r) for r in prior] + res
    holm(ext, key="holm_genarm11")
    for r in res:
        r["p_holm_genarm11"] = next(e["p_holm_genarm11"] for e in ext if e["id"] == r["id"])
        r["pass_holm_genarm11"] = r["p_holm_genarm11"] < 0.05
    genarm11 = {e["id"]: {"p_tost": e["p_tost"], "p_holm": e["p_holm_genarm11"],
                          "pass": e["pass_holm_genarm11"]} for e in ext}

    args.out.write_text(json.dumps({"delta": 0.05, "alpha": 0.05,
                                    "multiplicity": "Holm within the 4 nofn claims "
                                    "(pass_holm_nofn); Holm within genarm 7 + nofn 4 "
                                    "(pass_holm_genarm11)",
                                    "claims": res, "genarm11": genarm11}, indent=2) + "\n")
    for r in res:
        print(f"{r['id']:22s} est={r['estimate']:+.4f} "
              f"95=[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] repro={r['reproduces_published']} "
              f"90=[{r['ci90'][0]:+.4f},{r['ci90'][1]:+.4f}] p={r['p_tost']:.4f} "
              f"pass={r['pass_unadjusted']} holm4={r['p_holm_nofn']:.4f} "
              f"holm11={r['p_holm_genarm11']:.4f}")
    print("genarm11:", json.dumps(genarm11, indent=1))


if __name__ == "__main__":
    main()
