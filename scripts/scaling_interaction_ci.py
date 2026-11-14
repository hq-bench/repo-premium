#!/usr/bin/env python3
"""C5: paired within-family size effects and their cross-cell interaction.

For each encoder family (small -> large) at canonical c=1000 stripped
chunking (canon202607/<model>_<cell>_c1000), compute per query the uplift
  u(q) = Delta_large(q) - Delta_small(q)
which equals dense_large - dense_small because BM25 is encoder-independent
(checked). Report mean uplift per cell with a hierarchical bootstrap
(repositories, then queries within repository; paired within query), and the
interaction uplift_B - uplift_A, bootstrapping the two cells independently
(they share repositories but no queries; the repo-clustered variant resamples
the same repository draw for both cells, preserving shared-repo dependence).
"""
from paths import DATA_ROOT
import json
from pathlib import Path
import numpy as np

ROOT = DATA_ROOT
CANON = ROOT / "smoke_test_results" / "canon202607"
FAMILIES = {
    "jina-code": ("jina05", "jina15"),
    "C2LLM": ("c2llm05", "c2llm7"),
    "Qwen3-Emb": ("qwen06", "qwen8"),
    "Qwen3-Emb(0.6->4)": ("qwen06", "qwen4"),
    "E5": ("e5base", "e5large"),
    "F2LLM": ("f2llm06", "f2llm4"),
    "gte-Qwen2": ("gteq15", "gteq7"),
}
N_BOOT, SEED = 10_000, 0


def load(model, cell):
    out = {}
    for f in sorted((CANON / f"{model}_{cell}_c1000").glob("*/smoke_test_django.jsonl")):
        for r in map(json.loads, f.open()):
            out[r["query_id"]] = (f.parent.name, r["bm25_ndcg10"], r["dense_ndcg10"])
    return out


def cell_uplift(small, large, cell):
    S, L = load(small, cell), load(large, cell)
    common = sorted(set(S) & set(L))
    bm_diff = max(abs(S[q][1] - L[q][1]) for q in common)
    rows = {}
    for q in common:
        repo = S[q][0]
        rows.setdefault(repo, []).append(
            ((L[q][2] - L[q][1]) - (S[q][2] - S[q][1]), S[q][2] - S[q][1], L[q][2] - L[q][1]))
    return rows, len(common), bm_diff


def draw(rows, repos, rng):
    vals = []
    for p in repos:
        v = rows.get(p)
        if v:
            idx = rng.integers(0, len(v), len(v))
            vals.extend(v[i][0] for i in idx)
    return np.mean(vals) if vals else np.nan


def main():
    rng = np.random.default_rng(SEED)
    res = {}
    for fam, (s, l) in FAMILIES.items():
        A, nA, bA = cell_uplift(s, l, "cellA")
        B, nB, bB = cell_uplift(s, l, "cellB")
        flat = lambda R, k: np.mean([x[k] for v in R.values() for x in v])
        pt = {c: (flat(R, 1), flat(R, 2), flat(R, 0)) for c, R in (("A", A), ("B", B))}
        repos = sorted(set(A) | set(B))
        bA_, bB_, bI = [], [], []
        for _ in range(N_BOOT):
            # shared repository draw for both cells (cluster on repo)
            rd = list(rng.choice(repos, len(repos)))
            ua, ub = draw(A, rd, rng), draw(B, rd, rng)
            bA_.append(ua); bB_.append(ub); bI.append(ub - ua)
        ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
        res[fam] = dict(nA=nA, nB=nB, bm25_maxdiff=[bA, bB],
                        A=dict(small=pt["A"][0], large=pt["A"][1], uplift=pt["A"][2], ci=ci(bA_)),
                        B=dict(small=pt["B"][0], large=pt["B"][1], uplift=pt["B"][2], ci=ci(bB_)),
                        interaction=dict(mean=pt["B"][2] - pt["A"][2], ci=ci(bI)))
        r = res[fam]
        print(f"{fam:20s} nA={nA} nB={nB} bm25diff={bA:.0e}/{bB:.0e}")
        for c in "AB":
            print(f"   cell{c}: {r[c]['small']:+.3f} -> {r[c]['large']:+.3f}  uplift {r[c]['uplift']:+.3f} "
                  f"[{r[c]['ci'][0]:+.3f},{r[c]['ci'][1]:+.3f}]")
        print(f"   B-A interaction {r['interaction']['mean']:+.3f} "
              f"[{r['interaction']['ci'][0]:+.3f},{r['interaction']['ci'][1]:+.3f}]")
    out = ROOT / "analysis" / "scaling_interaction_ci.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
