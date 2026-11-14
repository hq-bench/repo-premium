#!/usr/bin/env python3
"""Score the frozen decision model v2 (serve decision) on the LOCKBOX — once.

Reads the frozen artifact (refuses to run if its sha256 differs from the
spec), the lockbox scores from qpp_lockbox.py, and grep top-10 lists from the
pools' cached canonical runs (grep does not depend on the dense encoder).
Features are built with the same functions as on dev. The evaluation follows
analysis/decision_model_v2_artifacts/PREREGISTRATION.txt.

Stage `guard` checks loaders only (dense top-10 identical to the cached
canonical lists; feature completeness) and prints no relevance metric.
Stage `score` computes the preregistered statistics and writes
analysis/decision_model_v2_lockbox.json. It refuses to overwrite an existing
result.
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from decision_model import ndcg10, p_features, rrf  # noqa: E402
from decision_model_v2 import FROZEN, N_BOOT, read_jsonl, score_feats  # noqa: E402
from router_ceiling import features as q_features  # noqa: E402

ROOT = _HERE.parent
DATA = DATA_ROOT
LOCKBOX = DATA / "smoke_test_results/qpp_lockbox202610"
OUT = DATA / "analysis/decision_model_v2_lockbox.json"
CACHED = {"r9_ts": DATA / "smoke_test_results/r9b_ts_c1000",
          "r9_go": DATA / "smoke_test_results/r9c_go_c1000",
          "r9_rust": DATA / "smoke_test_results/r9d_rust_c1000",
          "deepswe": DATA / "deepswe_pull/retrieval/jina-0.5b"}


def cached_run(pool: str, key: str) -> dict:
    f = CACHED[pool] / key / "smoke_test_django.jsonl"
    return {r["query_id"]: r for r in read_jsonl(f)} if f.is_file() else {}


def repo_of() -> dict:
    out = {}
    for line in (DATA / "deepswe_pull/task_repo.tsv").read_text().splitlines():
        if line.strip():
            task, repo = line.split()[:2]
            out[task] = repo
    return out


def build_rows():
    task_repo = repo_of()
    rows, guard = [], defaultdict(int)
    for f in sorted(LOCKBOX.glob("*.jsonl")):
        for r in read_jsonl(f):
            ref = cached_run(r["pool"], r["key"]).get(r["query_id"])
            if ref is None:
                guard["no_cached_run"] += 1
                continue
            d = [p for p, _ in r["dense"][:10]]
            b = [p for p, _ in r["bm25"][:10]]
            guard["n"] += 1
            guard["dense_identical"] += d == ref["dense_top10"]
            g = ref.get("grep_top10", [])
            repo = task_repo.get(r["key"], r["key"]) if r["pool"] == "deepswe" else r["key"]
            gold = r["gold_files"]
            row = {"pool": r["pool"], "repo": repo, "key": r["key"], "query_id": r["query_id"],
                   "dense": ndcg10(d, gold), "bm25": ndcg10(b, gold),
                   "rrf_db": ndcg10(rrf(d, b), gold)}
            row.update(q_features(r["query"], r["tau_hat"]))
            row.update(p_features(r["query"], d, b, g))
            row.update(score_feats("bm25", r["bm25"], r["bm25_mean"], r["bm25_std"]))
            row.update(score_feats("dense", r["dense"], r["dense_mean"], r["dense_std"]))
            rows.append(row)
    return rows, dict(guard)


def load_frozen():
    spec = json.loads((FROZEN / "serve_spec.json").read_text())
    blob = (FROZEN / "serve_gbt_QPS.pkl").read_bytes()
    if hashlib.sha256(blob).hexdigest() != spec["sha256"]:
        raise SystemExit("frozen artifact hash mismatch — refusing to score")
    return spec, pickle.loads(blob)


def boot(rows, vec, seed=0):
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["repo"]].append(i)
    groups = list(by.values())
    rng = np.random.default_rng(seed)
    vals = np.array([vec[[i for g in rng.integers(0, len(groups), len(groups)) for i in groups[g]]].mean()
                     for _ in range(N_BOOT)])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], float((vals <= 0).mean())


def stats(rows, routed):
    base = np.array([r["rrf_db"] for r in rows])
    dense = np.array([r["dense"] for r in rows])
    ci, p = boot(rows, routed - base)
    ci_ni, _ = boot(rows, base - dense)
    oracle = np.array([max(r[a] for a in ("dense", "bm25", "rrf_db")) for r in rows])
    return {"n": len(rows), "n_repos": len({r["repo"] for r in rows}),
            "always": {a: float(np.mean([r[a] for r in rows])) for a in ("dense", "bm25", "rrf_db")},
            "routed": float(routed.mean()),
            "routed_minus_base": float((routed - base).mean()), "ci": ci, "p_one_sided": p,
            "base_minus_dense": float((base - dense).mean()), "base_minus_dense_ci": ci_ni,
            "base_noninferior_at_-0.05": ci_ni[0] > -0.05,
            "oracle_minus_base": float((oracle - base).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["guard", "score"])
    args = ap.parse_args()
    spec, frozen = load_frozen()
    rows, guard = build_rows()
    missing = [f for f in spec["features"] if rows and f not in rows[0]]
    print("guard:", guard, "| missing features:", missing, "| frozen", spec["sha256"][:12])
    if args.stage == "guard" or missing:
        return 0 if not missing else 1
    if OUT.exists():
        raise SystemExit(f"{OUT} exists — the lockbox is scored once")
    X = np.array([[r[f] for f in spec["features"]] for r in rows], dtype=float)
    P = np.column_stack([frozen["models"][a].predict(X) for a in spec["actions"]])
    j = P.argmax(1)
    routed = np.array([r[spec["actions"][jj]] if P[i, jj] > 0 else r["rrf_db"]
                       for i, (r, jj) in enumerate(zip(rows, j))])
    res = {"frozen_sha256": spec["sha256"], "guard": guard, "pooled": stats(rows, routed),
           "deviation_rate": float(np.mean(P.max(1) > 0)), "by_pool": {}}
    for pool in sorted({r["pool"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["pool"] == pool]
        res["by_pool"][pool] = stats([rows[i] for i in idx], routed[idx])
    res["success"] = res["pooled"]["ci"][0] > 0
    excl = {"arktype-json-schema-refs-dependencies", "fd-deterministic-multi-key-sorting",
            "valibot-recursive-schema-composition"}  # ADDENDUM_before_scoring.txt item 3
    keep = [i for i, r in enumerate(rows) if r.get("key") not in excl]
    res["sensitivity_excl_3"] = stats([rows[i] for i in keep], routed[keep])
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
