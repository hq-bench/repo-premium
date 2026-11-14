#!/usr/bin/env python3
"""Metric robustness for the headline pipelines: nDCG@10, Recall@10, MRR@10.

Per cell (A, B, RA, RB), from the cached top-10 lists:
  dense      canonical jina-code-0.5b, c=1000
  bm25       BM25 over whole files (BM25's tuned chunk size)
  gen        BM25 over whole files on the query plus generated anchors
  rrf_db     RRF (k=60) of dense and bm25
Reports means and the dense-minus-bm25 and dense-minus-gen differences per
metric with repository-cluster bootstrap CIs (10,000, seed 20260813).
"""
from paths import DATA_ROOT
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_model import rrf  # noqa: E402
from genarm_rebase import DENSE, WHOLE, control_whole, gen_dir, load  # noqa: E402


def ndcg(lst, gold):
    g = set(gold)
    dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(lst[:10]) if x in g)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(len(g), 10)))


def recall(lst, gold):
    return len(set(lst[:10]) & set(gold)) / len(gold)


def mrr(lst, gold):
    return next((1 / (i + 1) for i, x in enumerate(lst[:10]) if x in set(gold)), 0.0)


METRICS = {"nDCG@10": ndcg, "Recall@10": recall, "MRR@10": mrr}


def boot(by, n=10000, seed=20260813):
    rng = random.Random(seed)
    g = list(by.values())
    d = sorted(sum(s) / len(s) for s in ([x for _ in g for x in g[rng.randrange(len(g))]] for _ in range(n)))
    return [d[int(.025 * n)], d[int(.975 * n)]]


res = {}
for cell in ("A", "B", "RA", "RB"):
    dn, bw, gw = load(DENSE[cell]), control_whole(cell), load(gen_dir(cell, WHOLE))
    keys = sorted(set(dn) & set(bw) & set(gw))
    lists = {k: {"dense": dn[k]["dense_top10"], "bm25": bw[k]["bm25_top10"], "gen": gw[k]["bm25_top10"],
                 "rrf_db": rrf(dn[k]["dense_top10"], bw[k]["bm25_top10"])} for k in keys}
    res[cell] = {}
    for mname, fn in METRICS.items():
        vals = {k: {a: fn(l, dn[k]["gold_files"]) for a, l in lists[k].items()} for k in keys}
        out = {"means": {a: sum(v[a] for v in vals.values()) / len(keys) for a in ("dense", "bm25", "gen", "rrf_db")}}
        for name, (x, y) in {"dense_minus_bm25": ("dense", "bm25"), "dense_minus_gen": ("dense", "gen"),
                             "rrf_db_minus_dense": ("rrf_db", "dense")}.items():
            by = defaultdict(list)
            for k in keys:
                by[k[0]].append(vals[k][x] - vals[k][y])
            out[name] = {"mean": sum(v for g in by.values() for v in g) / len(keys), "ci95": boot(by)}
        res[cell][mname] = out
        m = out["means"]
        print(f"{cell:2s} {mname:9s} dense {m['dense']:.3f} bm25 {m['bm25']:.3f} gen {m['gen']:.3f} rrf {m['rrf_db']:.3f} | "
              + " ".join(f"{n} {out[n]['mean']:+.3f} [{out[n]['ci95'][0]:+.3f},{out[n]['ci95'][1]:+.3f}]"
                         for n in ("dense_minus_bm25", "dense_minus_gen", "rrf_db_minus_dense")))
(DATA_ROOT / 'analysis/metrics_robustness.json').write_text(json.dumps(res, indent=2) + "\n")
