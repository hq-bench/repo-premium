"""tau statistics quoted in the text (tab:tau-by-cell, LORO range, tau=0 strip, R-OOD quartiles).

Same conventions as plot_hero_phase_diagram.py: delta = stored delta_ndcg10 (c=1000), tau = tau_top1,
quartiles by np.quantile + digitize, permutation p over 10,000 shuffles (seed 20260704).
Run from the staging root (acl_taufix) after run_taufix.sh <mode>; reads the tau files in place.
"""
from paths import DATA_ROOT
import glob
import json
import sys

import numpy as np
from scipy import stats

SM = str(DATA_ROOT / 'smoke_test_results')
POOLS = {"cellA": (f"{SM}/canon202607/jina05_cellA_c1000", f"{SM}/tau_variants_c1000/cellA"),
         "cellB": (f"{SM}/canon202607/jina05_cellB_c1000", f"{SM}/tau_variants_c1000/cellB"),
         "heldout": (f"{SM}/heldout_jina05_c1000", f"{SM}/tau_variants_law/heldout")}


def rows(pool):
    rdir, tdir = POOLS[pool]
    tau = {json.loads(l)["query_id"]: json.loads(l)["tau_top1"]
           for f in glob.glob(f"{tdir}/*/tau_variants.jsonl") for l in open(f) if l.strip()}
    out = []
    for f in sorted(glob.glob(f"{rdir}/*/smoke_test_django.jsonl")):
        repo = f.split("/")[-2]
        for l in open(f):
            r = json.loads(l)
            if r["query_id"] in tau:
                out.append((repo, tau[r["query_id"]], r["delta_ndcg10"]))
    return out


def describe(rs, name, perm=True):
    pos = [(rp, t, d) for rp, t, d in rs if t > 0]
    zero = [d for _, t, d in rs if t == 0]
    x = np.array([t for _, t, _ in pos]); y = np.array([d for _, _, d in pos])
    r, _ = stats.pearsonr(x, y)
    p = float("nan")
    if perm:
        rng = np.random.default_rng(20260704)
        p = float((np.abs([stats.pearsonr(x, rng.permutation(y))[0] for _ in range(10000)]) >= abs(r)).mean())
    e = np.quantile(x, [0, .25, .5, .75, 1]); b = np.clip(np.digitize(x, e[1:-1]), 0, 3)
    q = [float(y[b == i].mean()) for i in range(4)]
    mono = all(q[i] >= q[i + 1] for i in range(3))
    print(f"{name:10s} n={len(rs):4d} n_tau>0={len(pos):4d} r={r:+.3f} p_perm={p:.3f} "
          f"Q={[round(v, 3) for v in q]} mono={mono} | tau=0 n={len(zero)} mean={np.mean(zero) if zero else float('nan'):+.3f}"
          f" | mean all={np.mean([d for *_, d in rs]):+.3f}")
    return pos


def main():
    A, B, H = rows("cellA"), rows("cellB"), rows("heldout")
    describe(A, "cellA"); describe(B, "cellB"); pos = describe(A + B, "pooled")
    repos = sorted({rp for rp, *_ in pos})
    loro = [stats.pearsonr([t for rp, t, _ in pos if rp != k], [d for rp, _, d in pos if rp != k])[0] for k in repos]
    print(f"LORO r range [{min(loro):+.3f}, {max(loro):+.3f}] over {len(repos)} drops")
    describe(H, "heldout", perm=False)


if __name__ == "__main__":
    sys.exit(main())
