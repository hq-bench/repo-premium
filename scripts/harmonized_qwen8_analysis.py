"""Second encoder (Qwen3-Embedding-8B) on the harmonized raw pools.

premium_tuned = Qwen8 dense (c=1000) - whole-file BM25; residual = Qwen8 dense - whole-file BM25 on the query
plus generated anchors; repository-block 95% CIs. Guard: on the original Raw-Post queries the Qwen8 top-10
must reproduce the cached canonical run (canon202607/qwen8_cellB_c1000). Run from the work dir.
"""
from paths import DATA_ROOT
import glob
import json
import math
import random

WHOLE = 1000000000
CANON_Q8 = str(DATA_ROOT / 'smoke_test_results/canon202607/qwen8_cellB_c1000')
POOLS = {"Raw-Post": ("cellB", "out", "qrels_harmonized/cellB", "out/qwen8_shared_c1000"),
         "Raw-Pre'23": ("cellRA", "out24", "qrels_h24/cellRA", "out24/qwen8_cellRA_c1000"),
         "Raw-Pre'22": ("cellRB", "out", "qrels_harmonized/cellRB", "out/qwen8_shared_c1000")}


def load(d):
    return {json.loads(l)["query_id"]: dict(json.loads(l), _repo=f.split("/")[-2])
            for f in glob.glob(d + "/*/smoke_test_django.jsonl") for l in open(f) if l.strip()}


def ndcg(t, g):
    g = set(g)
    return sum(1 / math.log2(i + 2) for i, f in enumerate(t[:10]) if f in g) / \
        sum(1 / math.log2(i + 2) for i in range(min(len(g), 10)))


def ci(rows):
    rng, g = random.Random(20261015), {}
    for rp, v in rows:
        g.setdefault(rp, []).append(v)
    ks, d = sorted(g), []
    for _ in range(10000):
        s = [x for _ in ks for x in g[ks[rng.randrange(len(ks))]]]
        d.append(sum(s) / len(s))
    d.sort()
    return [d[250], d[9750]]


def main():
    out = {}
    canon = load(CANON_Q8)
    for name, (cell, o, qd, qdir) in POOLS.items():
        ids = {json.loads(l)["query_id"] for f in glob.glob(qd + "/*.jsonl") for l in open(f)}
        q8, bw, gw = load(qdir), load(f"{o}/bm25_{cell}_c{WHOLE}"), load(f"{o}/gen_{cell}_c{WHOLE}")
        rows = []
        for q in sorted(ids):
            if q not in q8 or q not in bw or q not in gw:
                continue
            g = q8[q]["gold_files"]
            d = ndcg(q8[q]["dense_top10"], g)
            rows.append((q8[q]["_repo"], d, ndcg(bw[q]["bm25_top10"], g), ndcg(gw[q]["bm25_top10"], g), q))
        n = len(rows)
        res = {"n": n, "n_expected": len(ids), "dense": sum(x[1] for x in rows) / n}
        for k, f in (("premium_tuned", lambda x: x[1] - x[2]), ("residual_gen", lambda x: x[1] - x[3])):
            v = [(x[0], f(x)) for x in rows]
            res[k] = {"mean": sum(y for _, y in v) / n, "ci95": ci(v)}
        if cell == "cellB":
            sh = [x[4] for x in rows if x[4] in canon]
            res["guard"] = {"compared": len(sh),
                            "identical": sum(q8[q]["dense_top10"] == canon[q]["dense_top10"] for q in sh)}
        out[name] = res
        print(f"{name:11s} n={n}/{len(ids)} dense {res['dense']:.3f} | premium_tuned "
              f"{res['premium_tuned']['mean']:+.3f} {[round(x, 3) for x in res['premium_tuned']['ci95']]} | residual "
              f"{res['residual_gen']['mean']:+.3f} {[round(x, 3) for x in res['residual_gen']['ci95']]}"
              + (f" | guard {res['guard']}" if "guard" in res else ""))
    json.dump(out, open("harmonized_qwen8.json", "w"), indent=1)


if __name__ == "__main__":
    main()
