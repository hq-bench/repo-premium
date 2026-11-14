#!/usr/bin/env python3
"""C1: unpaired CIs for the R-AMBIG low/high gold-share subgroup contrasts.

Joins anchor_ambiguity/<cell>/<repo>.jsonl with the canonical per-query cache
(canon202607/jina05_<cell>_c1000) and, within each tau_top1 quartile, splits
queries at the median gold_share. Reports the (high - low) difference in
premium and in BM25 nDCG@10 with (i) a query-level two-sample bootstrap and
(ii) a repository-clustered two-sample bootstrap (resample repos, then
queries within repo, keeping each repo's low and high members together).
"""
from paths import DATA_ROOT
import json, sys
from pathlib import Path
import numpy as np

ROOT = DATA_ROOT
PULL = ROOT / "smoke_test_results"
N_BOOT, SEED = 10_000, 0


def load(cell):
    rows = []
    for f in sorted((ROOT / "anchor_ambiguity" / cell).glob("*.jsonl")):
        repo = f.stem
        amb = {r["query_id"]: r for r in map(json.loads, f.open())}
        rf = PULL / "canon202607" / f"jina05_{cell}_c1000" / repo / "smoke_test_django.jsonl"
        res = {r["query_id"]: r for r in map(json.loads, rf.open())}
        # tau from the ambiguity file itself (matches canonical tau_top1 exactly)
        tau = {q: a["tau_top1"] for q, a in amb.items()}
        for qid, a in amb.items():
            if qid in res and qid in tau:
                r = res[qid]
                rows.append(dict(repo=repo, tau=tau[qid], share=a["gold_share"],
                                 bm25=r["bm25_ndcg10"], dense=r["dense_ndcg10"]))
    return rows


def boot(lo, hi, key, clustered, rng):
    def stat(L, H):
        return np.mean([r[key] for r in H]) - np.mean([r[key] for r in L])
    point = stat(lo, hi)
    out = np.empty(N_BOOT)
    if not clustered:
        lo_a, hi_a = np.array([r[key] for r in lo]), np.array([r[key] for r in hi])
        for b in range(N_BOOT):
            out[b] = rng.choice(hi_a, len(hi_a)).mean() - rng.choice(lo_a, len(lo_a)).mean()
    else:
        repos = sorted({r["repo"] for r in lo + hi})
        by = {p: ([r for r in lo if r["repo"] == p], [r for r in hi if r["repo"] == p]) for p in repos}
        for b in range(N_BOOT):
            L, H = [], []
            for p in rng.choice(repos, len(repos)):
                l, h = by[p]
                if l: L += [l[i] for i in rng.integers(0, len(l), len(l))]
                if h: H += [h[i] for i in rng.integers(0, len(h), len(h))]
            out[b] = stat(L, H) if L and H else np.nan
    lo_ci, hi_ci = np.nanpercentile(out, [2.5, 97.5])
    return point, lo_ci, hi_ci


def main():
    rng = np.random.default_rng(SEED)
    result = {}
    for cell in ("cellA", "cellB"):
        rows = load(cell)
        for r in rows:
            r["delta"] = r["dense"] - r["bm25"]
        taus = np.array([r["tau"] for r in rows])
        edges = np.quantile(taus, [0, .25, .5, .75, 1])
        q = np.clip(np.searchsorted(edges, taus, side="right") - 1, 0, 3)
        print(f"== {cell}: n={len(rows)}")
        result[cell] = []
        for k in range(4):
            sub = [r for r, qq in zip(rows, q) if qq == k]
            med = np.median([r["share"] for r in sub])
            lo = [r for r in sub if r["share"] <= med]
            hi = [r for r in sub if r["share"] > med]
            if not lo or not hi:
                print(f"  Q{k+1}: degenerate split (median share {med})"); continue
            rec = dict(quartile=k + 1, n_lo=len(lo), n_hi=len(hi), median_share=med)
            for key in ("delta", "bm25"):
                for cl in (False, True):
                    p, a, b = boot(lo, hi, key, cl, rng)
                    rec[f"{key}_{'repo' if cl else 'query'}"] = [p, a, b]
            rec["bm25_lo"], rec["bm25_hi"] = np.mean([r["bm25"] for r in lo]), np.mean([r["bm25"] for r in hi])
            result[cell].append(rec)
            print(f"  Q{k+1} n={len(lo)}/{len(hi)} bm25 {rec['bm25_lo']:.3f}->{rec['bm25_hi']:.3f} "
                  f"dPrem={rec['delta_query'][0]:+.3f} q[{rec['delta_query'][1]:+.3f},{rec['delta_query'][2]:+.3f}] "
                  f"repo[{rec['delta_repo'][1]:+.3f},{rec['delta_repo'][2]:+.3f}]")
    out = ROOT / "analysis" / "ambig_subgroup_ci.json"
    out.write_text(json.dumps(result, indent=1, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
