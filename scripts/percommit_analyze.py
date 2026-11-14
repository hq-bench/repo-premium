#!/usr/bin/env python3
"""Premiums with per-task pre-change snapshots vs the shared snapshot (percommit_retrieval.py output).

For each pool: dense - BM25 (chunk, c=1000) and dense - whole-file BM25 nDCG@10, on queries whose
gold survives in the per-commit corpus, with a repo-clustered bootstrap CI; the same queries'
shared-snapshot premiums (canonical runs, indexed labels) for comparison. Guard: guard-* pools
must equal the canonical runs.
"""
from __future__ import annotations
from paths import DATA_ROOT

import json
import math
import random
from pathlib import Path

D = DATA_ROOT
SM = (DATA_ROOT / 'smoke_test_results')
SHARED = {"Cur-Pre": ("canon202607/jina05_cellA_c1000", "bm25_sweep202610/bm25_cellA_c1000000000"),
          "Raw-Post": ("canon202607/jina05_cellB_c1000", "bm25_sweep202610/bm25_cellB_c1000000000"),
          "Raw-Pre'23": ("r8_robust_jina-0.5b_c1000", "gen_sweep202610/control_cellRA_c1000000000"),
          "Raw-Pre'22": ("r8_2022_jina-0.5b_c1000", "gen_sweep202610/control_cellRB_c1000000000")}


def load(d):
    out = {}
    for f in (SM / d).glob("*/smoke_test_django.jsonl"):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l); out[r["query_id"]] = r
    return out


def rci(rows):
    g = {}
    for rp, v in rows:
        g.setdefault(rp, []).append(v)
    ks = sorted(g); rng = random.Random(20261011); d = []
    for _ in range(10000):
        s = [v for _ in ks for v in g[ks[rng.randrange(len(ks))]]]
        d.append(sum(s) / len(s))
    d.sort()
    return [round(d[250], 3), round(d[9750], 3)]


def main():
    recs = [json.loads(l) for f in sorted((D / "percommit202610/out").glob("*/results.jsonl"))
            for l in f.read_text().splitlines() if l.strip()]
    res = {"guard": {}, "pools": {}}
    for gp, (dd, bd) in [("guard-Cur-Pre", SHARED["Cur-Pre"]), ("guard-Raw-Post", SHARED["Raw-Post"])]:
        dn, bm = load(dd), load(bd)
        rs = [r for r in recs if r["pool"] == gp]
        res["guard"][gp] = {"n": len(rs),
                            "dense_mismatch": sum(r["dense_top10"] != dn[r["query_id"]]["dense_top10"] for r in rs),
                            "bm25_mismatch": sum(r["bm25_top10"] != dn[r["query_id"]]["bm25_top10"] for r in rs),
                            "bm25w_mismatch": sum(r["bm25w_top10"] != bm[r["query_id"]]["bm25_top10"] for r in rs)}
    for pool, (dd, bd) in SHARED.items():
        dn, bm = load(dd), load(bd)
        rs = [r for r in recs if r["pool"] == pool]
        ok = [r for r in rs if r["gold_files"]]
        row = {"n_tasks": len(rs), "n_scored": len(ok),
               "n_gold_partly_missing": sum(bool(r["gold_missing"]) for r in rs),
               "median_corpus_files": sorted(r["n_corpus_files"] for r in rs)[len(rs) // 2] if rs else None}
        for k, lab in (("bm25", "chunk"), ("bm25w", "whole")):
            pc = [(r["repo"], r["dense_ndcg10"] - r[f"{k}_ndcg10"]) for r in ok]
            row[f"percommit_{lab}"] = {"premium": round(sum(v for _, v in pc) / len(pc), 3), "ci95": rci(pc)} if pc else None
        # shared-snapshot premium on the same queries (stored indexed metrics)
        sh_c = [(r["repo"], dn[r["query_id"]]["dense_ndcg10"] - dn[r["query_id"]]["bm25_ndcg10"]) for r in ok if r["query_id"] in dn]
        sh_w = [(r["repo"], dn[r["query_id"]]["dense_ndcg10"] - bm[r["query_id"]]["bm25_ndcg10"]) for r in ok if r["query_id"] in dn and r["query_id"] in bm]
        row["shared_chunk"] = {"n": len(sh_c), "premium": round(sum(v for _, v in sh_c) / len(sh_c), 3), "ci95": rci(sh_c)} if sh_c else None
        row["shared_whole"] = {"n": len(sh_w), "premium": round(sum(v for _, v in sh_w) / len(sh_w), 3), "ci95": rci(sh_w)} if sh_w else None
        row["dense_ndcg_percommit"] = round(sum(r["dense_ndcg10"] for r in ok) / max(len(ok), 1), 3)
        row["bm25w_ndcg_percommit"] = round(sum(r["bm25w_ndcg10"] for r in ok) / max(len(ok), 1), 3)
        res["pools"][pool] = row
    print(json.dumps(res, indent=1))
    (D / "analysis/indexed/percommit_premium.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
