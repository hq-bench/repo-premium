"""Generation test on the held-out pool (PRs merged >= 2025-04-01; the generator qwen35-122B was
released 2026-02-24): dense vs whole-file BM25 on the query and on the
query plus generated anchors, source-file labels, repository-block and query-level 95% CIs. The pool
was mined from 2025-04-01, so the exposure-controlled subset keeps PRs created after the release.

Run from the work dir (heldout_gen) after run_heldout_gen.sh. Dense is the cached canonical run.
"""
from paths import DATA_ROOT
import glob
import json
import math
import random

import build_source_only_mirror as M

CANON = str(DATA_ROOT / 'smoke_test_results/heldout_jina05_c1000')
WHOLE = 1000000000


def load(d):
    return {json.loads(l)["query_id"]: dict(json.loads(l), _repo=f.split("/")[-2])
            for f in glob.glob(d + "/*/smoke_test_django.jsonl") for l in open(f) if l.strip()}


def src_gold(gold):
    return [g for g in gold if g.endswith(".py") and not M.is_test(g)]


def ndcg(t, g):
    g = set(g)
    return sum(1 / math.log2(i + 2) for i, f in enumerate(t[:10]) if f in g) / \
        sum(1 / math.log2(i + 2) for i in range(min(len(g), 10)))


def cis(rows):
    rng, g = random.Random(20261015), {}
    for rp, v in rows:
        g.setdefault(rp, []).append(v)
    ks, vals = sorted(g), [v for _, v in rows]
    rb, qb = [], []
    for _ in range(10000):
        s = [x for _ in ks for x in g[ks[rng.randrange(len(ks))]]]
        rb.append(sum(s) / len(s))
        s = [vals[rng.randrange(len(vals))] for _ in vals]
        qb.append(sum(s) / len(s))
    rb.sort(); qb.sort()
    return [rb[250], rb[9750]], [qb[250], qb[9750]]


QRELS = str(DATA_ROOT / 'qrels_heldout')
RELEASE = "2026-02-25"  # qwen35-122B released 2026-02-24 (Epoch AI); the subset keeps PRs CREATED after it


def main():
    meta = {json.loads(l)["query_id"]: json.loads(l) for f in glob.glob(QRELS + "/*.jsonl") for l in open(f) if l.strip()}
    dense_all, out = load(CANON), {}
    for subset in ("all", "created_after_release"):
      dense = {q: r for q, r in dense_all.items()
               if subset == "all" or meta[q].get("created_at", "") >= RELEASE}
      for C in (1000, WHOLE):
        plain, gen = load(f"out/plain_c{C}"), load(f"out/gen_c{C}")
        rows = []
        for q, r in dense.items():
            gold = src_gold(r["gold_files"])
            if not gold or q not in plain or q not in gen:
                continue
            d, b, e = ndcg(r["dense_top10"], gold), ndcg(plain[q]["bm25_top10"], gold), ndcg(gen[q]["bm25_top10"], gold)
            rows.append({"repo": r["_repo"], "dense": d, "bm25": b, "gen": e})
        n = len(rows)
        res = {"n": n, "n_repos": len({x["repo"] for x in rows}),
               **{k: sum(x[k] for x in rows) / n for k in ("dense", "bm25", "gen")}}
        for name, f in [("premium", lambda x: x["dense"] - x["bm25"]), ("dense_minus_gen", lambda x: x["dense"] - x["gen"]),
                        ("gen_gain", lambda x: x["gen"] - x["bm25"])]:
            v = [(x["repo"], f(x)) for x in rows]
            rb, qb = cis(v)
            res[name] = {"mean": sum(y for _, y in v) / n, "ci_repo": rb, "ci_query": qb}
        out[f"{subset}_c{C}"] = res
        print(f"== {subset} BM25 c={C}: n={n} repos={res['n_repos']} dense {res['dense']:.3f} bm25 {res['bm25']:.3f} gen {res['gen']:.3f}")
        for k in ("premium", "dense_minus_gen", "gen_gain"):
            x = res[k]
            print(f"   {k:16s} {x['mean']:+.3f} repo[{x['ci_repo'][0]:+.3f},{x['ci_repo'][1]:+.3f}] "
                  f"query[{x['ci_query'][0]:+.3f},{x['ci_query'][1]:+.3f}]")
    json.dump(out, open("heldout_gen_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
