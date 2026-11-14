"""RM3 control on the harmonized raw pools (rm3_bm25.run_repo, unchanged settings).

Guard: RM3's first pass must reproduce the harmonized whole-file BM25 top-10 (out*/bm25_<cell>_c1e9);
tied-score reorders of an identical set are counted, anything else fails. Reports RM3 - BM25 and
generate-then-grep - RM3 per pool with repository-block 95% CIs. Run from the work dir.
"""
import glob
import json
import math
import os
import random

WHOLE = 1000000000
OUT = {"B": "out", "RA": "out24", "RB": "out"}
NAME = {"B": "Raw-Post", "RA": "Raw-Pre'23", "RB": "Raw-Pre'22"}


def load(d):
    return {json.loads(l)["query_id"]: json.loads(l) for f in glob.glob(d + "/*/smoke_test_django.jsonl")
            for l in open(f) if l.strip()}


def ndcg(t, g):
    g = set(g)
    return sum(1 / math.log2(i + 2) for i, f in enumerate(t[:10]) if f in g) / \
        sum(1 / math.log2(i + 2) for i in range(min(len(g), 10)))


def ci(rows):
    rng, g = random.Random(20260813), {}
    for rp, v in rows:
        g.setdefault(rp, []).append(v)
    ks, d = sorted(g), []
    for _ in range(10000):
        s = [x for _ in ks for x in g[ks[rng.randrange(len(ks))]]]
        d.append(sum(s) / len(s))
    d.sort()
    return [d[250], d[9750]]


def main():
    res, ok = {}, True
    for cell in ("B", "RA", "RB"):
        bm = load(f"{OUT[cell]}/bm25_cell{cell}_c{WHOLE}")
        gen = load(f"{OUT[cell]}/gen_cell{cell}_c{WHOLE}")
        rows, same, ties, bad = [], 0, 0, []
        for f in glob.glob(f"out/rm3/{cell}__*.jsonl"):
            repo = os.path.basename(f)[len(cell) + 2:-6]
            for l in open(f):
                r = json.loads(l)
                q = r["query_id"]
                if q not in bm:
                    continue
                if r["bm25_top10"] == bm[q]["bm25_top10"]:
                    same += 1
                elif set(r["bm25_top10"]) == set(bm[q]["bm25_top10"]):
                    ties += 1
                else:
                    bad.append(q)
                g = r["gold_files"]
                rows.append((repo, ndcg(bm[q]["bm25_top10"], g), ndcg(r["rm3_top10"], g), ndcg(gen[q]["bm25_top10"], g)))
        n = len(rows)
        out = {"n": n, "guard": {"identical": same, "tie_reorders": ties, "different": len(bad)},
               "bm25": sum(x[1] for x in rows) / n, "rm3": sum(x[2] for x in rows) / n, "gen": sum(x[3] for x in rows) / n}
        for k, f in (("rm3_minus_bm25", lambda x: x[2] - x[1]), ("gen_minus_rm3", lambda x: x[3] - x[2])):
            v = [(x[0], f(x)) for x in rows]
            out[k] = {"mean": sum(y for _, y in v) / n, "ci95": ci(v)}
        res[NAME[cell]] = out
        ok = ok and not bad
        print(f"{NAME[cell]:11s} n={n} guard {out['guard']} bm25 {out['bm25']:.3f} rm3 {out['rm3']:.3f} gen {out['gen']:.3f} | "
              f"rm3-bm25 {out['rm3_minus_bm25']['mean']:+.3f} {[round(x, 3) for x in out['rm3_minus_bm25']['ci95']]} | "
              f"gen-rm3 {out['gen_minus_rm3']['mean']:+.3f} {[round(x, 3) for x in out['gen_minus_rm3']['ci95']]}")
        if bad:
            print("   guard differences:", bad[:5])
    json.dump(res, open("harmonized_rm3.json", "w"), indent=1)
    print("GUARD", "OK" if ok else "FAILED")


if __name__ == "__main__":
    main()
