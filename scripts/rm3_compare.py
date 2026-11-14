#!/usr/bin/env python3
"""Compare RM3 (rm3_bm25.py) with whole-file BM25, generate-then-grep and dense.

Guard: RM3's first pass must reproduce the whole-file BM25 top-10 lists
(bm25_sweep202610 for A/B, gen_sweep202610 control arm for RA/RB); tied-score
reorders of an identical top-10 set are counted, anything else fails.
Repository-cluster bootstrap CIs (10,000, seed 20260813).
"""
from paths import DATA_ROOT
import glob
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from genarm_rebase import DENSE, WHOLE, control_whole, gen_dir, load  # noqa: E402

RM3 = (DATA_ROOT / 'smoke_test_results/rm3_202610')


def boot(by, n=10000, seed=20260813):
    rng = random.Random(seed)
    g = list(by.values())
    d = []
    for _ in range(n):
        s = []
        for _ in g:
            s.extend(g[rng.randrange(len(g))])
        d.append(sum(s) / len(s))
    d.sort()
    return [d[int(.025 * n)], d[int(.975 * n)]]


res = {}
for cell in ("A", "B", "RA", "RB"):
    rm = {}
    for f in glob.glob(str(RM3 / f"{cell}__*.jsonl")):
        repo = os.path.basename(f)[len(cell) + 2:-6]
        for line in open(f):
            r = json.loads(line)
            rm[(repo, r["query_id"])] = r
    cw, gw, dn = control_whole(cell), load(gen_dir(cell, WHOLE)), load(DENSE[cell])
    keys = sorted(set(rm) & set(cw) & set(gw) & set(dn))
    off = [k for k in keys if rm[k]["bm25_top10"] != cw[k]["bm25_top10"]]
    real = [k for k in off if set(rm[k]["bm25_top10"]) != set(cw[k]["bm25_top10"])]
    by = defaultdict(lambda: defaultdict(list))
    for k in keys:
        d = dn[k]["dense_ndcg10"]
        by["rm3_minus_bm25"][k[0]].append(rm[k]["rm3_ndcg10"] - cw[k]["bm25_ndcg10"])
        by["premium_vs_rm3"][k[0]].append(d - rm[k]["rm3_ndcg10"])
        by["gen_minus_rm3"][k[0]].append(gw[k]["bm25_ndcg10"] - rm[k]["rm3_ndcg10"])
    mean = lambda key: sum(v for g in by[key].values() for v in g) / len(keys)
    res[cell] = {"n": len(keys), "guard_tie_reorders": len(off) - len(real), "guard_real_diffs": len(real),
                 "means": {"bm25_whole": sum(cw[k]["bm25_ndcg10"] for k in keys) / len(keys),
                           "rm3": sum(rm[k]["rm3_ndcg10"] for k in keys) / len(keys),
                           "gen_whole": sum(gw[k]["bm25_ndcg10"] for k in keys) / len(keys),
                           "dense": sum(dn[k]["dense_ndcg10"] for k in keys) / len(keys)},
                 **{key: {"mean": mean(key), "ci95": boot(by[key])} for key in by}}
    c = res[cell]
    print(f"{cell:2s} n={c['n']} guard real={c['guard_real_diffs']} ties={c['guard_tie_reorders']} | "
          + " ".join(f"{k}={v:.3f}" for k, v in c["means"].items()))
    for key in by:
        print(f"     {key:15s} {c[key]['mean']:+.3f} [{c[key]['ci95'][0]:+.3f},{c[key]['ci95'][1]:+.3f}]")
(DATA_ROOT / 'analysis/rm3_vs_genarm.json').write_text(json.dumps(res, indent=2) + "\n")
