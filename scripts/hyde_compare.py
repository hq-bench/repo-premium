#!/usr/bin/env python3
"""Does generation help dense retrieval as much as BM25? (hyde_dense.py output)

Per cell, with both lexical arms on whole files (BM25's tuned chunk size):
  dense_gain    dense(expanded query) - dense(original query)
  bm25_gain     BM25(expanded) - BM25(original)
  gain_gap      bm25_gain - dense_gain   (> 0: anchors help BM25 more)
  premium_both  dense(expanded) - BM25(expanded): the premium when both get the anchors
Guard: dense(original) top-10 must equal the cached canonical dense run.
Repository-cluster bootstrap (10,000, seed 20260813).
"""
from paths import DATA_ROOT
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_model import ndcg10  # noqa: E402
from genarm_rebase import DENSE, WHOLE, control_whole, gen_dir, load  # noqa: E402

HYDE = (DATA_ROOT / 'smoke_test_results/hyde_202610')


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
    hy = {}
    for f in HYDE.glob(f"{cell}__*.jsonl"):
        repo = f.stem[len(cell) + 2:]
        for line in f.read_text().splitlines():
            r = json.loads(line)
            hy[(repo, r["query_id"])] = r
    dn, cw, gw = load(DENSE[cell]), control_whole(cell), load(gen_dir(cell, WHOLE))
    keys = sorted(set(hy) & set(dn) & set(cw) & set(gw))
    guard = sum(hy[k]["orig_top10"] == dn[k]["dense_top10"] for k in keys)
    by = defaultdict(lambda: defaultdict(list))
    means = defaultdict(float)
    for k in keys:
        gold = hy[k]["gold_files"]
        d0, d1, dg = (ndcg10(hy[k][f], gold) for f in ("orig_top10", "expanded_top10", "generated_top10"))
        b0, b1 = cw[k]["bm25_ndcg10"], gw[k]["bm25_ndcg10"]
        for name, v in (("dense_gain", d1 - d0), ("bm25_gain", b1 - b0), ("gain_gap", (b1 - b0) - (d1 - d0)),
                        ("premium_both", d1 - b1), ("dense_anchors_only_gain", dg - d0)):
            by[name][k[0]].append(v)
        for name, v in (("dense_orig", d0), ("dense_expanded", d1), ("dense_anchors_only", dg),
                        ("bm25_orig", b0), ("bm25_expanded", b1)):
            means[name] += v / len(keys)
    res[cell] = {"n": len(keys), "guard_dense_orig_identical": guard, "means": dict(means),
                 **{n: {"mean": sum(v for g in by[n].values() for v in g) / len(keys), "ci95": boot(by[n])}
                    for n in by}}
    c = res[cell]
    print(f"{cell:2s} n={c['n']} guard {guard}/{len(keys)} | " + " ".join(f"{k}={v:.3f}" for k, v in means.items()))
    for n in by:
        print(f"     {n:24s} {c[n]['mean']:+.3f} [{c[n]['ci95'][0]:+.3f},{c[n]['ci95'][1]:+.3f}]")
(DATA_ROOT / 'analysis/hyde_compare.json').write_text(json.dumps(res, indent=2) + "\n")
