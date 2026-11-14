"""Numbers for the paper edit that harmonized_analysis.py does not produce (harmonized raw pools).

  * premium under labels as released (every file the change touched, "as mined" in tab:r8);
  * premium under "all edited" labels (indexed .py files the change edited, tests kept) at c=1000
    (query-level CI, tab:r8 "as mined") and against whole-file BM25 (repo-block CI);
  * share of queries whose top-1 file is a test, for dense and whole-file BM25;
  * share of edited files that are tests / not indexed, and of queries containing them;
  * Raw-Post minus Cur-Pre gap at c=1000 (source and all-edited labels), query-level and
    repository-block CIs as in law_clustered_bootstrap.cross_cell_gap;
  * the quality-filter funnel under the harmonized gold rule.
Every statistic is also printed on the original-pool queries (orig_selected), to check against
the paper. Run from the work dir (harmonized202610).
"""
from paths import DATA_ROOT
import glob
import json
import math
from pathlib import Path

import numpy as np

from build_harmonized_qrels import PY_REPOS, harmonized_gold, tree
from build_source_only_mirror import is_test

D = DATA_ROOT
WHOLE = 1000000000
POOLS = {  # name: (mined qrel dir, snapshot, harmonized qrels, outputs, dense dir)
    "Raw-Post": ("qrels_post_cutoff", "repos", "qrels_harmonized/cellB", "out", "out/dense_all_c1000", "cellB"),
    "Raw-Pre'23": ("qrels_r5", "repos_2024q1", "qrels_h24/cellRA", "out24", "out24/dense_cellRA_c1000", "cellRA"),
    "Raw-Pre'22": ("qrels_r8_2022", "repos", "qrels_harmonized/cellRB", "out", "out/dense_all_c1000", "cellRB"),
}
TEXT = ["body_too_short", "too_many_files", "docs_only", "ci_only", "tests_only", "no_code_files"]


def load(d):
    return {json.loads(l)["query_id"]: dict(json.loads(l), _repo=f.split("/")[-2])
            for f in glob.glob(f"{d}/*/smoke_test_django.jsonl") for l in open(f) if l.strip()}


def ndcg(t, g):
    g = set(g)
    return sum(1 / math.log2(i + 2) for i, f in enumerate(t[:10]) if f in g) / \
        sum(1 / math.log2(i + 2) for i in range(min(len(g), 10)))


def ci_q(v, seed=20261010):
    rng, v = np.random.default_rng(seed), np.array(v)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(10000)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def ci_repo(rows, seed=20261010):
    rng, g = np.random.default_rng(seed), {}
    for rp, v in rows:
        g.setdefault(rp, []).append(v)
    ks = sorted(g); m = []
    for _ in range(10000):
        s = [x for k in rng.choice(ks, len(ks)) for x in g[k]]
        m.append(np.mean(s))
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def gap(a, b, seed=20261010):
    """b minus a; a, b = lists of (repo, delta). Query-level and repo-block (joint repo draw)."""
    rng = np.random.default_rng(seed)
    ya, yb = np.array([v for _, v in a]), np.array([v for _, v in b])
    ra, rb = np.array([r for r, _ in a]), np.array([r for r, _ in b])
    q = [yb[rng.integers(0, len(yb), len(yb))].mean() - ya[rng.integers(0, len(ya), len(ya))].mean()
         for _ in range(10000)]
    repos = np.unique(np.concatenate([ra, rb]))
    ia = {r: np.where(ra == r)[0] for r in repos}; ib = {r: np.where(rb == r)[0] for r in repos}
    rbk = []
    for _ in range(10000):
        pick = repos[rng.integers(0, len(repos), len(repos))]
        xa = [ya[rng.choice(ia[r], len(ia[r]))] for r in pick if len(ia[r])]
        xb = [yb[rng.choice(ib[r], len(ib[r]))] for r in pick if len(ib[r])]
        if xa and xb:
            rbk.append(np.concatenate(xb).mean() - np.concatenate(xa).mean())
    pc = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]
    return round(float(yb.mean() - ya.mean()), 4), pc(q), pc(rbk)


def main():
    A = load("_paper_indexed/canon/jina05_cellA_c1000")
    a_src = [(r["_repo"], r["dense_ndcg10"] - r["bm25_ndcg10"]) for r in A.values()]
    out = {}
    for name, (mq, snap, hq, o, dd, cell) in POOLS.items():
        meta = {json.loads(l)["query_id"]: json.loads(l) for f in glob.glob(f"{hq}/*.jsonl") for l in open(f)}
        dense, bc, bw = load(dd), load(f"{o}/bm25_{cell}_c1000"), load(f"{o}/bm25_{cell}_c{WHOLE}")
        trees = {}
        res = {}
        for sub in ("old", "all"):
            qs = [q for q in meta if q in dense and (sub == "all" or meta[q].get("orig_selected"))]
            rows = {"src_c": [], "all_c": [], "src_t": [], "all_t": [], "rel_c": [], "rel_t": []}
            top1 = {"dense": 0, "bm25w": 0}
            nfiles = ntest = nnonidx = qtest = qnonidx = 0
            for q in qs:
                m, rp = meta[q], dense[q]["_repo"]
                if rp not in trees:
                    trees[rp] = tree(D / snap / rp)
                files = trees[rp]
                orig = m["orig_gold_files"]
                py = [g[4:] if (g not in files and g.startswith("src/") and g[4:] in files) else g
                      for g in orig if g.endswith(".py")]
                allg = [g for g in py if g in files]
                nfiles += len(orig); t = sum(map(is_test, orig)); ntest += t; qtest += t > 0
                ni = sum(not g.endswith(".py") for g in orig); nnonidx += ni; qnonidx += ni > 0
                d, c, w = dense[q]["dense_top10"], bc[q]["bm25_top10"], bw[q]["bm25_top10"]
                src = m["gold_files"]
                rows["src_c"].append((rp, ndcg(d, src) - ndcg(c, src)))
                rows["all_c"].append((rp, ndcg(d, allg) - ndcg(c, allg)))
                rows["src_t"].append((rp, ndcg(d, src) - ndcg(w, src)))
                rows["all_t"].append((rp, ndcg(d, allg) - ndcg(w, allg)))
                rows["rel_c"].append((rp, ndcg(d, orig) - ndcg(c, orig)))  # labels as released
                rows["rel_t"].append((rp, ndcg(d, orig) - ndcg(w, orig)))
                top1["dense"] += bool(d) and is_test(d[0]); top1["bm25w"] += bool(w) and is_test(w[0])
            n = len(qs)
            r = {"n": n, **{k: round(float(np.mean([v for _, v in x])), 4) for k, x in rows.items()},
                 "src_c_ciq": [round(x, 3) for x in ci_q([v for _, v in rows["src_c"]])],
                 "all_c_ciq": [round(x, 3) for x in ci_q([v for _, v in rows["all_c"]])],
                 "src_t_cir": [round(x, 3) for x in ci_repo(rows["src_t"])],
                 "all_t_cir": [round(x, 3) for x in ci_repo(rows["all_t"])],
                 "rel_c_ciq": [round(x, 3) for x in ci_q([v for _, v in rows["rel_c"]])],
                 "top1_test_dense": round(top1["dense"] / n, 3), "top1_test_bm25w": round(top1["bm25w"] / n, 3),
                 "test_file_share": round(ntest / nfiles, 3), "queries_with_test": round(qtest / n, 3),
                 "nonindexed_file_share": round(nnonidx / nfiles, 3), "queries_with_nonindexed": round(qnonidx / n, 3)}
            if name == "Raw-Post":
                r["gap_src_c1000"] = gap(a_src, rows["src_c"])
                r["gap_all_c1000_vs_A_src"] = gap(a_src, rows["all_c"])
                r["gap_rel_c1000_vs_A_src"] = gap(a_src, rows["rel_c"])
            res[sub] = r
        # funnel under the harmonized gold rule
        f = {"candidates": 0, **{k: 0 for k in TEXT}, "other_text": 0, "no_source_gold": 0, "gt3": 0,
             "missing": 0, "kept": 0}
        for qf in sorted((D / mq).glob("*.jsonl")):
            if qf.name.startswith(".") or qf.stem.split("__")[1] not in PY_REPOS:
                continue
            files = trees.get(qf.stem) or tree(D / snap / qf.stem)
            for l in open(qf, encoding="utf-8"):
                if not l.strip():
                    continue
                q = json.loads(l); f["candidates"] += 1
                fr = q.get("filter_reason")
                if fr:
                    f[fr if fr in TEXT else "other_text"] += 1
                    continue
                g = harmonized_gold(q["gold_files"], files)
                if not g:
                    f["no_source_gold"] += 1
                elif len(g) > 3:
                    f["gt3"] += 1
                elif not all(x in files for x in g):
                    f["missing"] += 1
                else:
                    f["kept"] += 1
        res["funnel"] = f
        out[name] = res
        print(f"== {name}")
        for sub in ("old", "all"):
            print(f"  {sub}: {json.dumps(res[sub])}")
        print(f"  funnel: {json.dumps(f)}")
    Path("harmonized_extras.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
