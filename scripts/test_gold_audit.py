#!/usr/bin/env python3
"""Audit: test files in the gold sets of the core pools, and their effect on the premium.

The paper states that gold sets are non-test source files. The evaluated labels
contain test files in the raw pools (from the merged PR's changed-file list) but
not in the curated pool (SWE-bench keeps tests in a separate test_patch). This
script measures the asymmetry and re-scores nDCG@10 with test files removed from
the gold set, using each retriever's stored top-10 (the re-scorer reproduces
the stored nDCG@10 exactly, which is checked). Retrieved test files then count
as non-relevant. Queries whose gold set is test-only are dropped.

Outputs ${REPO_PREMIUM_DATA:-data}/analysis/test_gold_audit.json.
"""
from __future__ import annotations
from paths import DATA_ROOT

import json
import math
import random
import re
from pathlib import Path

SM = (DATA_ROOT / 'smoke_test_results')
DENSE = {"A": SM / "canon202607/jina05_cellA_c1000", "B": SM / "canon202607/jina05_cellB_c1000",
         "RA": SM / "r8_robust_jina-0.5b_c1000", "RB": SM / "r8_2022_jina-0.5b_c1000"}
TUNED = {"A": SM / "bm25_sweep202610/bm25_cellA_c1000000000", "B": SM / "bm25_sweep202610/bm25_cellB_c1000000000"}
TEST = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|/conftest\.py$")
SEED, NBOOT = 20260813, 10000


def is_test(f: str) -> bool:
    return bool(TEST.search(f))


def ndcg(top: list[str], gold: list[str]) -> float:
    g = set(gold)
    dcg = sum(1 / math.log2(i + 2) for i, f in enumerate(top[:10]) if f in g)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(len(g), 10)))


def load(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("*/smoke_test_django.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[(f.parent.name, r["query_id"])] = r
    return out


def repo_ci(by_repo: dict[str, list[float]]) -> list[float]:
    rng, reps, d = random.Random(SEED), sorted(by_repo), []
    for _ in range(NBOOT):
        s = [v for _ in reps for v in by_repo[reps[rng.randrange(len(reps))]]]
        d.append(sum(s) / len(s))
    d.sort()
    return [d[int(.025 * NBOOT)], d[int(.975 * NBOOT)]]


def main() -> int:
    res: dict = {"test_pattern": TEST.pattern, "pools": {}}
    for cell, d in DENSE.items():
        dn = load(d)
        tuned = load(TUNED[cell]) if cell in TUNED else None
        n = n_test_q = n_gold = n_test_g = n_test_only = 0
        maxdiff = 0.0
        lab, src = {}, {}
        lab_t, src_t = {}, {}
        for k, r in dn.items():
            gold = r["gold_files"]
            t = sum(map(is_test, gold))
            n += 1; n_gold += len(gold); n_test_g += t; n_test_q += t > 0; n_test_only += t == len(gold)
            maxdiff = max(maxdiff, abs(ndcg(r["bm25_top10"], gold) - r["bm25_ndcg10"]),
                          abs(ndcg(r["dense_top10"], gold) - r["dense_ndcg10"]))
            src_gold = [f for f in gold if not is_test(f)]
            if not src_gold:
                continue
            lab.setdefault(k[0], []).append(r["dense_ndcg10"] - r["bm25_ndcg10"])
            src.setdefault(k[0], []).append(ndcg(r["dense_top10"], src_gold) - ndcg(r["bm25_top10"], src_gold))
            if tuned and k in tuned:
                lab_t.setdefault(k[0], []).append(r["dense_ndcg10"] - tuned[k]["bm25_ndcg10"])
                src_t.setdefault(k[0], []).append(ndcg(r["dense_top10"], src_gold) - ndcg(tuned[k]["bm25_top10"], src_gold))
        if maxdiff > 1e-12:
            raise SystemExit(f"re-scorer does not reproduce stored nDCG for {cell}: {maxdiff}")
        mean = lambda b: sum(map(sum, b.values())) / sum(map(len, b.values()))
        p = {"queries": n, "queries_with_test_gold": n_test_q, "test_only_queries": n_test_only,
             "gold_files": n_gold, "test_gold_files": n_test_g,
             "premium_c1000_as_labeled": mean(lab), "premium_c1000_source_only": mean(src),
             "premium_c1000_source_only_ci95": repo_ci(src)}
        if tuned:
            p.update({"premium_tuned_as_labeled": mean(lab_t), "premium_tuned_source_only": mean(src_t),
                      "premium_tuned_source_only_ci95": repo_ci(src_t)})
        res["pools"][cell] = p
        print(cell, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}))
    (DATA_ROOT / 'analysis/test_gold_audit.json').write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
