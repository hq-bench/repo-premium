#!/usr/bin/env python3
"""Score the generation tier on the lockbox — once (PREREG_generation_tier.txt).

Inputs
  smoke_test_results/qpp_lockbox202610/   dense (c=1000) and whole-file BM25 top lists per query
  smoke_test_results/gen_lockbox_bm25/    whole-file BM25 on query + generated anchors
  deepswe_pull/task_repo.tsv              DeepSWE task -> repository (bootstrap cluster)

Pipelines: dense, bm25, gen, rrf_db = RRF(dense, bm25), rrf_dg = RRF(dense, gen), k=60, top-10.
Primary: rrf_dg - dense, pooled, repository-clustered bootstrap (10,000, seed 0); success iff the
95% lower bound > 0. Secondary: rrf_dg - rrf_db, gen - bm25, dense - gen; per pool.

Guards (stage `guard`, no metrics): every lockbox query has a gen run; the gen run indexed the
same number of files as the lockbox scores; generation failures after the retry are counted.
Stage `score` refuses to overwrite an existing result.
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_model import ndcg10, rrf  # noqa: E402

D = DATA_ROOT
LOCK = D / "smoke_test_results/qpp_lockbox202610"
GEN = D / "smoke_test_results/gen_lockbox_bm25"
GENQ = D / "smoke_test_results/gen_lockbox_qrels"
OUT = D / "analysis/gen_tier_lockbox.json"


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def rows():
    task_repo = dict(l.split()[:2] for l in (D / "deepswe_pull/task_repo.tsv").read_text().splitlines() if l.strip())
    out, guard = [], defaultdict(int)
    for f in sorted(LOCK.glob("*.jsonl")):
        pool, key = f.stem.split("__", 1)
        gdir = GEN / f"{pool}__{key}"
        gen = {r["query_id"]: r for r in read_jsonl(gdir / "smoke_test_django.jsonl")} if (gdir / "smoke_test_django.jsonl").is_file() else {}
        summ = json.loads((gdir / "smoke_test_django_summary.json").read_text()) if gen else {}
        genq = {r["query_id"]: r for r in read_jsonl(GENQ / pool / f"{key}.jsonl")} if (GENQ / pool / f"{key}.jsonl").is_file() else {}
        for r in read_jsonl(f):
            guard["n"] += 1
            g = gen.get(r["query_id"])
            if g is None:
                guard["missing_gen"] += 1
                continue
            guard["corpus_mismatch"] += int(summ.get("n_corpus_files") != r["n_files"])
            q = genq.get(r["query_id"], {})
            guard["generation_failed"] += int("generation_error" in q)
            gold = r["gold_files"]
            d = [p for p, _ in r["dense"][:10]]
            b = [p for p, _ in r["bm25"][:10]]
            gt = g["bm25_top10"]
            out.append({"pool": pool, "repo": task_repo.get(key, key) if pool == "deepswe" else key,
                        "dense": ndcg10(d, gold), "bm25": ndcg10(b, gold), "gen": ndcg10(gt, gold),
                        "rrf_db": ndcg10(rrf(d, b), gold), "rrf_dg": ndcg10(rrf(d, gt), gold)})
    return out, dict(guard)


def boot(rs, vec, seed=0, n=10000):
    by = defaultdict(list)
    for i, r in enumerate(rs):
        by[r["repo"]].append(i)
    groups = list(by.values())
    rng = np.random.default_rng(seed)
    vals = np.array([vec[[i for g in rng.integers(0, len(groups), len(groups)) for i in groups[g]]].mean()
                     for _ in range(n)])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], float((vals <= 0).mean())


def summarize(rs):
    arr = {k: np.array([r[k] for r in rs]) for k in ("dense", "bm25", "gen", "rrf_db", "rrf_dg")}
    res = {"n": len(rs), "n_repos": len({r["repo"] for r in rs}), "means": {k: float(v.mean()) for k, v in arr.items()}}
    for name, (a, b) in {"rrf_dg_minus_dense": ("rrf_dg", "dense"), "rrf_dg_minus_rrf_db": ("rrf_dg", "rrf_db"),
                         "gen_minus_bm25": ("gen", "bm25"), "dense_minus_gen": ("dense", "gen")}.items():
        v = arr[a] - arr[b]
        ci, p = boot(rs, v)
        res[name] = {"mean": float(v.mean()), "ci95": ci, "p_one_sided_le0": p}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["guard", "score"])
    args = ap.parse_args()
    rs, guard = rows()
    print("guard:", guard)
    if args.stage == "guard":
        return 0
    if guard.get("missing_gen") or guard.get("corpus_mismatch"):
        raise SystemExit("guard failed — not scoring")
    if OUT.exists():
        raise SystemExit(f"{OUT} exists — the lockbox is scored once")
    res = {"guard": guard, "pooled": summarize(rs), "by_pool": {}}
    for pool in sorted({r["pool"] for r in rs}):
        res["by_pool"][pool] = summarize([r for r in rs if r["pool"] == pool])
    res["success"] = res["pooled"]["rrf_dg_minus_dense"]["ci95"][0] > 0
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
