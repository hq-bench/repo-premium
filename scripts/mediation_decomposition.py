#!/usr/bin/env python3
"""Mediation decomposition of the Cell A -> Cell B premium gap, plus
oracle-router ceiling.

Question 1 (mediation): the paper's stated mechanism for the
mining-methodology effect is "raw mining shifts query mass toward low tau,
where the dense premium is large". But Table tau-by-cell shows Cell B's
premium exceeds Cell A's at *every* tau quartile, so the tau-composition
shift cannot be the whole story. This script runs a DiNardo/Oaxaca-style
decomposition on the per-query cache:

    gap = E_B[Delta] - E_A[Delta]
        = (tau-composition term) + (within-tau structural term)
        [+ an out-of-support tau=0 term, since Cell A has no tau=0 queries]

Composition term: reweight Cell A's per-bin mean Delta by Cell B's tau-bin
weights (common support, tau>0). Structural term: B-weighted mean of the
per-bin (B - A) differences. tau=0 term: Cell-B-only mass outside A's
support, contributing p0 * (Delta_B(tau=0) - counterfactual).

Question 2 (oracle ceiling): per-query max(bm25, dense) vs dense alone --
the headroom available to a perfect router, overall and by tau quartile.

Reads <pull>/<cellA dir>/*/smoke_test_django.jsonl (same cache the hero
figure uses) plus the matching tau_variants tree, and writes
<pull>/mediation_decomposition.json.

The original ${REPO_PREMIUM_DATA:-data}/coreb_pull scratch cache is transient; the durable mirror
lives in the repo. The CANONICAL invocation -- the c1000 pipeline the paper's
tables report (Delta_A=+0.121, Delta_B=+0.206) -- is

    python mediation_decomposition.py \
        --pull ${REPO_PREMIUM_DATA:-data}/smoke_test_results \
        --cell-a canon202607/jina05_cellA_c1000 \
        --cell-b canon202607/jina05_cellB_c1000 \
        --tau tau_variants_c1000

reproducing composition +0.008 (~6% of the tau>0 gap of +0.135) and
within-tau structural +0.127 (~94%).

The older cellA_stripped x cellB_stripped_c1000 pair also runs, but it is a
DIFFERENT Cell A (Delta_A=+0.095, a chunk-size difference) whose terms do not
match tab:cells. Do not mix the two pipelines in one reported decomposition.
"""
from paths import DATA_ROOT

import argparse
import json
from pathlib import Path

import numpy as np

PULL = (DATA_ROOT / 'smoke_test_results')
CELL_DIRS = {"cellA": "cellA", "cellB": "cellB"}
TAU_DIR = "tau_variants"
N_BOOT = 10_000
SEED = 0


def load_cell(cell):
    # tau_top1 from tau_variants.jsonl is the paper-canonical MaxIDF tau
    # (see plot_hero_phase_diagram.py); tau_q in the smoke files is stale.
    rows = []
    for repo_dir in sorted((PULL / CELL_DIRS[cell]).iterdir()):
        if not repo_dir.is_dir():
            continue
        f = repo_dir / "smoke_test_django.jsonl"
        if not f.exists():
            continue
        with f.open() as fh:
            recs = {r["query_id"]: r
                    for r in (json.loads(l) for l in fh)}
        tv_file = PULL / TAU_DIR / cell / repo_dir.name / "tau_variants.jsonl"
        if not tv_file.exists():
            continue
        with tv_file.open() as fh:
            for line in fh:
                tv = json.loads(line)
                r = recs.get(tv["query_id"])
                if r is None:
                    continue
                rows.append(
                    (tv["tau_top1"], r["bm25_ndcg10"], r["dense_ndcg10"])
                )
    a = np.array(rows)
    return {"tau": a[:, 0], "bm25": a[:, 1], "dense": a[:, 2],
            "delta": a[:, 2] - a[:, 1]}


def bin_edges_pooled(tau_a, tau_b, n_bins):
    pooled = np.concatenate([tau_a[tau_a > 0], tau_b[tau_b > 0]])
    qs = np.quantile(pooled, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    return qs


def decompose(A, B, edges):
    """Return dict of decomposition terms for one (A, B) draw."""
    a_pos, b_pos = A["tau"] > 0, B["tau"] > 0
    p0 = float((~b_pos).mean())  # Cell B tau=0 mass (A has none)

    ba = np.digitize(A["tau"][a_pos], edges) - 1
    bb = np.digitize(B["tau"][b_pos], edges) - 1
    n_bins = len(edges) - 1
    wa = np.array([(ba == i).mean() for i in range(n_bins)])
    wb = np.array([(bb == i).mean() for i in range(n_bins)])
    ma = np.array([A["delta"][a_pos][ba == i].mean() if (ba == i).any()
                   else np.nan for i in range(n_bins)])
    mb = np.array([B["delta"][b_pos][bb == i].mean() if (bb == i).any()
                   else np.nan for i in range(n_bins)])

    d_a = float(A["delta"].mean())
    d_b = float(B["delta"].mean())
    d_a_pos = float(A["delta"][a_pos].mean())
    d_b_pos = float(B["delta"][b_pos].mean())
    d_b_zero = float(B["delta"][~b_pos].mean()) if p0 > 0 else 0.0

    # common-support (tau>0) decomposition
    ok = ~np.isnan(ma) & ~np.isnan(mb)
    counterfactual = float(np.nansum(np.where(ok, wb * ma, 0))
                           / np.where(ok, wb, 0).sum())
    composition = counterfactual - d_a_pos
    structural = d_b_pos - counterfactual
    # overall-gap accounting: gap = (1-p0)*(gap on tau>0) + p0*(...)
    gap = d_b - d_a
    gap_pos = d_b_pos - d_a_pos
    tau0_term = p0 * (d_b_zero - d_a)  # mass moved out of A's support

    return {
        "delta_A": d_a, "delta_B": d_b, "gap": gap,
        "delta_A_taupos": d_a_pos, "delta_B_taupos": d_b_pos,
        "gap_taupos": gap_pos,
        "composition_taupos": composition,
        "structural_taupos": structural,
        "p_tau0_B": p0, "delta_B_tau0": d_b_zero, "tau0_term": tau0_term,
        "bin_means_A": ma.tolist(), "bin_means_B": mb.tolist(),
        "bin_weights_A": wa.tolist(), "bin_weights_B": wb.tolist(),
    }


def bootstrap(A, B, edges, keys):
    rng = np.random.default_rng(SEED)
    na, nb = len(A["tau"]), len(B["tau"])
    out = {k: [] for k in keys}
    for _ in range(N_BOOT):
        ia = rng.integers(0, na, na)
        ib = rng.integers(0, nb, nb)
        Ar = {k: v[ia] for k, v in A.items()}
        Br = {k: v[ib] for k, v in B.items()}
        try:
            d = decompose(Ar, Br, edges)
        except (ZeroDivisionError, FloatingPointError):
            continue
        for k in keys:
            out[k].append(d[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in out.items()}


def oracle(C, edges):
    orc = np.maximum(C["bm25"], C["dense"])
    res = {
        "dense_mean": float(C["dense"].mean()),
        "bm25_mean": float(C["bm25"].mean()),
        "oracle_mean": float(orc.mean()),
        "headroom": float((orc - C["dense"]).mean()),
        "pct_bm25_wins": float((C["bm25"] > C["dense"]).mean()),
    }
    pos = C["tau"] > 0
    bins = np.digitize(C["tau"][pos], edges) - 1
    res["headroom_by_quartile"] = [
        float((orc[pos] - C["dense"][pos])[bins == i].mean())
        for i in range(len(edges) - 1)]
    res["pct_bm25_wins_by_quartile"] = [
        float((C["bm25"][pos] > C["dense"][pos])[bins == i].mean())
        for i in range(len(edges) - 1)]
    return res


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pull", type=Path, default=PULL,
                    help="root holding the per-cell caches and the tau tree")
    ap.add_argument("--cell-a", default="cellA",
                    help="subdirectory under --pull holding Cell A smoke files")
    ap.add_argument("--cell-b", default="cellB",
                    help="subdirectory under --pull holding Cell B smoke files")
    ap.add_argument("--tau", default="tau_variants",
                    help="subdirectory under --pull holding the tau_variants tree")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default <pull>/mediation_decomposition.json)")
    return ap.parse_args()


def main():
    global PULL, CELL_DIRS, TAU_DIR
    args = parse_args()
    PULL = args.pull
    CELL_DIRS = {"cellA": args.cell_a, "cellB": args.cell_b}
    TAU_DIR = args.tau

    A, B = load_cell("cellA"), load_cell("cellB")
    print(f"loaded: Cell A n={len(A['tau'])}, Cell B n={len(B['tau'])}")

    for n_bins, tag in [(4, "quartile"), (10, "decile")]:
        edges = bin_edges_pooled(A["tau"], B["tau"], n_bins)
        d = decompose(A, B, edges)
        keys = ["gap", "gap_taupos", "composition_taupos",
                "structural_taupos", "tau0_term"]
        cis = bootstrap(A, B, edges, keys)
        print(f"\n=== {tag} bins (tau>0 common support) ===")
        print(f"Delta_A = {d['delta_A']:+.3f}   Delta_B = {d['delta_B']:+.3f}"
              f"   overall gap = {d['gap']:+.3f} CI {cis['gap']}")
        print(f"tau>0 only: Delta_A = {d['delta_A_taupos']:+.3f}"
              f"  Delta_B = {d['delta_B_taupos']:+.3f}"
              f"  gap = {d['gap_taupos']:+.3f} CI {cis['gap_taupos']}")
        print(f"  composition (tau-shift) = {d['composition_taupos']:+.3f}"
              f" CI {cis['composition_taupos']}"
              f"  ({100*d['composition_taupos']/d['gap_taupos']:.0f}% of tau>0 gap)")
        print(f"  structural (within-tau) = {d['structural_taupos']:+.3f}"
              f" CI {cis['structural_taupos']}"
              f"  ({100*d['structural_taupos']/d['gap_taupos']:.0f}% of tau>0 gap)")
        print(f"  tau=0 out-of-support: p0(B) = {d['p_tau0_B']:.3f},"
              f" Delta_B(tau=0) = {d['delta_B_tau0']:+.3f},"
              f" term = {d['tau0_term']:+.3f} CI {cis['tau0_term']}")
        if n_bins == 4:
            print(f"  bin means A: {[round(x,3) for x in d['bin_means_A']]}")
            print(f"  bin means B: {[round(x,3) for x in d['bin_means_B']]}")
            print(f"  bin weights A: {[round(x,3) for x in d['bin_weights_A']]}")
            print(f"  bin weights B: {[round(x,3) for x in d['bin_weights_B']]}")
            result = {"decomposition": d, "cis": cis}

    edges4 = bin_edges_pooled(A["tau"], B["tau"], 4)
    print("\n=== oracle router ceiling (per-query max(bm25, dense)) ===")
    for name, C in [("Cell A", A), ("Cell B", B)]:
        o = oracle(C, edges4)
        print(f"{name}: dense {o['dense_mean']:.3f}  bm25 {o['bm25_mean']:.3f}"
              f"  oracle {o['oracle_mean']:.3f}"
              f"  headroom +{o['headroom']:.3f}"
              f"  bm25-wins {100*o['pct_bm25_wins']:.1f}%")
        print(f"  headroom by tau quartile (Q1..Q4): "
              f"{[round(x,3) for x in o['headroom_by_quartile']]}")
        print(f"  bm25-wins by tau quartile: "
              f"{[round(100*x,1) for x in o['pct_bm25_wins_by_quartile']]} %")
        result[f"oracle_{name.replace(' ', '')}"] = o

    out = args.out or (PULL / "mediation_decomposition.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
