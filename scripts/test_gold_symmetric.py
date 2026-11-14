#!/usr/bin/env python3
"""Symmetric gold-label check for the curated-vs-raw contrast.

Raw gold sets keep the test files the merged PR edited; Cur-Pre gold sets (SWE-bench
gold patch) have none. This script scores both arms under two symmetric definitions:

  source  -- test files removed from every gold set (the paper's stated task)
  edited  -- every file the resolving change edited; Cur-Pre gains the test files of its
             SWE-bench test_patch that exist at the pinned snapshot (raw pools required
             all gold files to exist, so the same rule applies)

nDCG@10 is re-scored from each retriever's stored top-10 (exact: nDCG@10 needs only the
top-10 and the gold-set size). Reports per-pool premia and the curated-raw gap with
repository-block bootstrap CIs (repos resampled jointly across the two pools).

Inputs: ${REPO_PREMIUM_DATA:-data}/analysis/swebv_test_files.json (instance_id -> test_patch files),
clones under ${REPO_PREMIUM_DATA:-data}/repos (the corpus snapshot).
Outputs ${REPO_PREMIUM_DATA:-data}/analysis/test_gold_symmetric.json.
"""
from __future__ import annotations
from paths import DATA_ROOT

import json
import random
from pathlib import Path

from build_source_only_mirror import is_test  # the paper's test-file rule (all languages)
from test_gold_audit import DENSE, TUNED, load, ndcg

CLONES = DATA_ROOT / "repos"
TESTS = json.loads((DATA_ROOT / 'analysis/swebv_test_files.json').read_text())
SEED, NBOOT = 20261010, 10000


def gold_sets(cell: str, repo: str, r: dict) -> dict[str, list[str]]:
    gold = r["gold_files"]
    src = [f for f in gold if not is_test(f)]
    if cell == "A":
        extra = [f for f in TESTS[r["query_id"]] if (CLONES / repo / f).is_file() and f not in gold]
        edited = gold + extra
    else:
        edited = gold
    return {"source": src, "edited": edited}


def premia(cell: str, bm25: dict) -> dict[str, dict[str, list[float]]]:
    """definition -> repo -> per-query premium (dense minus the given BM25 run)."""
    dn = load(DENSE[cell])
    out: dict = {"source": {}, "edited": {}, "_n_added_tests": 0, "_q_added": 0}
    for k, r in dn.items():
        if k not in bm25:
            continue
        g = gold_sets(cell, k[0], r)
        if cell == "A":
            add = len(g["edited"]) - len(r["gold_files"])
            out["_n_added_tests"] += add
            out["_q_added"] += add > 0
        for name, gs in g.items():
            if gs:
                out[name].setdefault(k[0], []).append(ndcg(r["dense_top10"], gs) - ndcg(bm25[k]["bm25_top10"], gs))
    return out


def mean(b: dict[str, list[float]]) -> float:
    return sum(map(sum, b.values())) / sum(map(len, b.values()))


def gap_ci(a: dict, b: dict) -> list[float]:
    rng, reps, d = random.Random(SEED), sorted(set(a) | set(b)), []
    for _ in range(NBOOT):
        pick = [reps[rng.randrange(len(reps))] for _ in reps]
        xa = [v for p in pick for v in a.get(p, [])]
        xb = [v for p in pick for v in b.get(p, [])]
        if xa and xb:
            d.append(sum(xb) / len(xb) - sum(xa) / len(xa))
    d.sort()
    return [d[int(.025 * len(d))], d[int(.975 * len(d))]]


def pool_ci(b: dict) -> list[float]:
    rng, reps, d = random.Random(SEED), sorted(b), []
    for _ in range(NBOOT):
        s = [v for _ in reps for v in b[reps[rng.randrange(len(reps))]]]
        d.append(sum(s) / len(s))
    d.sort()
    return [d[int(.025 * NBOOT)], d[int(.975 * NBOOT)]]


def main() -> int:
    res: dict = {}
    for proto, runs in {"c1000": {c: DENSE[c] for c in DENSE}, "tuned": TUNED}.items():
        P = {}
        for cell in runs:
            bm = load(runs[cell]) if proto == "tuned" else load(DENSE[cell])
            P[cell] = premia(cell, bm)
        block = {"added_tests_curated": {"files": P["A"]["_n_added_tests"], "queries": P["A"]["_q_added"]}}
        for name in ("source", "edited"):
            block[name] = {c: {"premium": mean(P[c][name]), "ci95": pool_ci(P[c][name]),
                               "n": sum(map(len, P[c][name].values()))} for c in P}
            block[name]["gap_B_minus_A"] = {"gap": mean(P["B"][name]) - mean(P["A"][name]),
                                            "ci95": gap_ci(P["A"][name], P["B"][name])}
        res[proto] = block
        print(proto, json.dumps(block, default=lambda x: round(x, 4))[:1500])
    (DATA_ROOT / 'analysis/test_gold_symmetric.json').write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
