"""Do the queries admitted by the harmonized gold rule behave differently from the original ones?

For each raw pool and pooled: mean of dense - expanded-query BM25 (whole files), the tuned premium and the
c=1000 premium on old vs newly admitted queries, with a repository-block bootstrap CI for new - old; the
same contrast within single-gold-file queries; and the covariates that differ (gold-set size, edited
files, stripped query length). Run from the work dir (harmonized202610) after harmonized_analysis.py.
"""
import glob
import json
import random

import numpy as np

from run_retrieval import strip_pr_boilerplate

POOLS = ("Raw-Post", "Raw-Pre'23", "Raw-Pre'22")
KEYS = ("dense_minus_genw", "prem_whole", "prem_c1000")


def load():
    rows = [json.loads(l) for l in open("harmonized_rows.jsonl")]
    meta = {}
    for d in ("qrels_harmonized/cellB", "qrels_h24/cellRA", "qrels_harmonized/cellRB"):
        for f in glob.glob(d + "/*.jsonl"):
            for l in open(f):
                r = json.loads(l)
                meta[r["query_id"]] = r
    for r in rows:
        m = meta[r["qid"]]
        r["ngold"], r["nedit"] = len(m["gold_files"]), m["orig_n_gold_files"]
        r["qlen"] = len(strip_pr_boilerplate(m["problem_statement"]))
    return rows


def gap_ci(rows, key, n_boot=4000):
    rng, reps, out = random.Random(1), sorted({r["repo"] for r in rows}), []
    by = {p: [r for r in rows if r["repo"] == p] for p in reps}
    for _ in range(n_boot):
        pick = [reps[rng.randrange(len(reps))] for _ in reps]
        o = [r[key] for p in pick for r in by[p] if r["old"]]
        n = [r[key] for p in pick for r in by[p] if not r["old"]]
        if o and n:
            out.append(np.mean(n) - np.mean(o))
    out.sort()
    return [round(out[int(.025 * len(out))], 3), round(out[int(.975 * len(out))], 3)]


def contrast(rows, key):
    o = [r[key] for r in rows if r["old"]]
    n = [r[key] for r in rows if not r["old"]]
    return {"old": float(np.mean(o)), "n_old": len(o), "new": float(np.mean(n)), "n_new": len(n),
            "diff": float(np.mean(n) - np.mean(o)), "ci_repo": gap_ci(rows, key)}


def main():
    rows, out = load(), {}
    for key in KEYS:
        out[key] = {}
        for p in POOLS + ("all",):
            rs = [r for r in rows if p == "all" or r["pool"] == p]
            c = out[key][p] = contrast(rs, key)
            print(f"{key:17s} {p:11s} old {c['old']:+.3f} (n{c['n_old']}) new {c['new']:+.3f} (n{c['n_new']}) "
                  f"new-old {c['diff']:+.3f} {c['ci_repo']}")
        c = out[key]["single_gold"] = contrast([r for r in rows if r["ngold"] == 1], key)
        print(f"{key:17s} single-gold old {c['old']:+.3f} (n{c['n_old']}) new {c['new']:+.3f} (n{c['n_new']}) "
              f"new-old {c['diff']:+.3f} {c['ci_repo']}")
    out["covariates"] = {k: {"old": float(np.mean([r[k] for r in rows if r["old"]])),
                             "new": float(np.mean([r[k] for r in rows if not r["old"]]))}
                         for k in ("ngold", "nedit", "qlen")}
    print("covariates", json.dumps(out["covariates"]))
    json.dump(out, open("harmonized_old_vs_new.json", "w"), indent=1)


if __name__ == "__main__":
    main()
