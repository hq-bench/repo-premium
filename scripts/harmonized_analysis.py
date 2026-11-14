"""Dense premium, fusion and LLM-expansion results on the harmonized raw pools.

Run from the harmonized work dir (${REPO_PREMIUM_DATA:-data}/harmonized202610).
Inputs (all per-repo smoke_test_django.jsonl):
  out/dense_all_c1000/            dense (jina-code-0.5b, c=1000, strip) over qrels_harmonized/all
  out/bm25_{pool}_c{C}/           BM25, C in {1000, whole}
  out/gen_{pool}_c{C}/            BM25 on the LLM-expanded query
Raw-Pre'23 is scored at the 2024-Q1 snapshot (qrels_h24/, out24/), as in the paper.
The old/new split comes from the qrels field orig_selected.

Metrics are recomputed from the stored top-10 lists. CIs: repository-block bootstrap (ci) and query-level (ci_query).
Outputs harmonized_analysis.json and per-query harmonized_rows.jsonl in the work dir.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

WHOLE = 1000000000
SEED, NBOOT = 20261010, 10000
RRF_K = 60
POOLS = {"cellB": "Raw-Post", "cellRA": "Raw-Pre'23", "cellRB": "Raw-Pre'22"}
# Raw-Pre'23 is scored at the 2024-Q1 snapshot (as in the paper's r8_robust runs): its own qrels and outputs.
QRELS = {"cellB": "qrels_harmonized/cellB", "cellRA": "qrels_h24/cellRA", "cellRB": "qrels_harmonized/cellRB"}
OUTS = {"cellB": "out", "cellRA": "out24", "cellRB": "out"}
DENSE = {"cellB": "out/dense_all_c1000", "cellRA": "out24/dense_cellRA_c1000", "cellRB": "out/dense_all_c1000"}


def load(d: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(d.glob("*/smoke_test_django.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["_repo"] = f.parent.name
                out[r["query_id"]] = r
    return out


def ndcg(top: list[str], gold: list[str]) -> float:
    g = set(gold)
    dcg = sum(1 / math.log2(i + 2) for i, f in enumerate(top[:10]) if f in g)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(len(g), 10)))


def recall(top: list[str], gold: list[str]) -> float:
    return len(set(top[:10]) & set(gold)) / len(set(gold))


def mrr(top: list[str], gold: list[str]) -> float:
    g = set(gold)
    return next((1 / (i + 1) for i, f in enumerate(top[:10]) if f in g), 0.0)


def rrf(a: list[str], b: list[str], k: int = RRF_K) -> list[str]:
    score: dict[str, float] = {}
    first: dict[str, tuple[int, int]] = {}
    for li, lst in enumerate((a, b)):
        for r, x in enumerate(lst[:10]):
            score[x] = score.get(x, 0.0) + 1.0 / (k + r + 1)
            first.setdefault(x, (r, li))
    return sorted(score, key=lambda x: (-score[x], first[x]))[:10]


def ci(rows: list[tuple[str, float]]) -> list[float]:
    rng, d = random.Random(SEED), []
    groups: dict[str, list[float]] = {}
    for rep, v in rows:
        groups.setdefault(rep, []).append(v)
    keys = sorted(groups)
    for _ in range(NBOOT):
        s = [v for _ in keys for v in groups[keys[rng.randrange(len(keys))]]]
        d.append(sum(s) / len(s))
    d.sort()
    return [d[int(.025 * NBOOT)], d[int(.975 * NBOOT)]]


def ci_query(vals: list[float]) -> list[float]:
    rng, d = random.Random(SEED), []
    for _ in range(NBOOT):
        s = [vals[rng.randrange(len(vals))] for _ in vals]
        d.append(sum(s) / len(s))
    d.sort()
    return [d[int(.025 * NBOOT)], d[int(.975 * NBOOT)]]


def summarize(rows: list[dict], key: str) -> dict:
    vals = [(r["repo"], r[key]) for r in rows]
    return {"mean": sum(v for _, v in vals) / len(vals), "ci": ci(vals), "ci_query": ci_query([v for _, v in vals])}


def main() -> None:
    out, all_rows = {}, []
    for pool, name in POOLS.items():
        meta = {}
        for f in sorted(Path(QRELS[pool]).glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    meta[r["query_id"]] = r
        dense, o = load(Path(DENSE[pool])), OUTS[pool]
        bw, bc = load(Path(f"{o}/bm25_{pool}_c{WHOLE}")), load(Path(f"{o}/bm25_{pool}_c1000"))
        gw, gc = load(Path(f"{o}/gen_{pool}_c{WHOLE}")), load(Path(f"{o}/gen_{pool}_c1000"))
        qids = sorted(q for q in bw if q in meta)
        missing = [q for q in qids if q not in dense or q not in bc or q not in gw or q not in gc]
        rows = []
        for q in qids:
            if q in missing:
                continue
            gold = meta[q]["gold_files"]
            assert sorted(gold) == sorted(bw[q]["gold_files"]) == sorted(dense[q]["gold_files"]), q
            L = {"dense": dense[q]["dense_top10"], "bm25_whole": bw[q]["bm25_top10"],
                 "bm25_c1000": bc[q]["bm25_top10"], "gen_whole": gw[q]["bm25_top10"],
                 "gen_c1000": gc[q]["bm25_top10"]}
            L["rrf_dense_bm25w"] = rrf(L["dense"], L["bm25_whole"])
            L["rrf_dense_genw"] = rrf(L["dense"], L["gen_whole"])
            r = {"qid": q, "repo": bw[q]["_repo"], "old": bool(meta[q].get("orig_selected")),
                 "tau_q": bc[q].get("tau_q")}
            for k, lst in L.items():
                r[f"{k}_ndcg"], r[f"{k}_r10"], r[f"{k}_mrr"] = ndcg(lst, gold), recall(lst, gold), mrr(lst, gold)
            r["prem_whole"] = r["dense_ndcg"] - r["bm25_whole_ndcg"]
            r["prem_c1000"] = r["dense_ndcg"] - r["bm25_c1000_ndcg"]
            r["dense_minus_genw"] = r["dense_ndcg"] - r["gen_whole_ndcg"]
            r["rrf_db_minus_dense"] = r["rrf_dense_bm25w_ndcg"] - r["dense_ndcg"]
            r["rrf_dg_minus_genw"] = r["rrf_dense_genw_ndcg"] - r["gen_whole_ndcg"]
            rows.append(r)
        all_rows += [dict(r, pool=name) for r in rows]
        keys = [k for k in rows[0] if k.endswith(("_ndcg", "_r10", "_mrr")) or k.startswith(("prem", "dense_minus", "rrf_d"))]
        res = {"n": len(rows), "n_repos": len({r["repo"] for r in rows}), "missing": missing}
        for sub, rs in [("all", rows), ("old", [r for r in rows if r["old"]]), ("new", [r for r in rows if not r["old"]])]:
            if rs:
                res[sub] = {"n": len(rs), **{k: summarize(rs, k) for k in keys}}
        out[name] = res
        a = res["all"]
        print(f"== {name}  n={res['n']} repos={res['n_repos']} missing={len(missing)}")
        for k in ["dense_ndcg", "bm25_whole_ndcg", "bm25_c1000_ndcg", "gen_whole_ndcg", "rrf_dense_bm25w_ndcg",
                  "rrf_dense_genw_ndcg", "prem_whole", "prem_c1000", "dense_minus_genw", "rrf_db_minus_dense",
                  "rrf_dg_minus_genw", "dense_mrr", "bm25_whole_mrr", "dense_r10", "bm25_whole_r10"]:
            o, n = res.get("old", {}).get(k), res.get("new", {}).get(k)
            print(f"  {k:22s} all {a[k]['mean']:+.3f} [{a[k]['ci'][0]:+.3f},{a[k]['ci'][1]:+.3f}]"
                  f"  old {o['mean'] if o else float('nan'):+.3f}  new {n['mean'] if n else float('nan'):+.3f}")
    Path("harmonized_analysis.json").write_text(json.dumps(out, indent=1))
    Path("harmonized_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_rows))


if __name__ == "__main__":
    main()
