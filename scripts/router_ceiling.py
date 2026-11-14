#!/usr/bin/env python3
"""router_ceiling.py — how much can per-query routing buy, and can a
query-only classifier capture it?

D0 (ceiling). For every pool with per-query dense and BM25 nDCG@10:
always-dense, always-BM25, the oracle per-query router, the headroom
(oracle - dense) and the share of exact ties (dense == BM25), which
nDCG@10's coarse grid makes common and which no router can exploit.
On the canonical jina-code-0.5b Cell A/B pools it also reproduces the
paper's R-ROUTER Cell B quartiles and the LOO hard-switch arm of
tab:router-hybrid as a loader check.

D1 (predictive test of "not identifiable from structural features").
The nine features of lexical_rescue_taxonomy.py minus the two that read
gold files (gold_basename, rare_identifier), plus query-only extras
(tau-hat, identifier flag, length). A logistic regression and a small
gradient-boosted tree predict "BM25 strictly beats dense" under
leave-one-repository-out CV; we report out-of-fold AUROC and the routed
nDCG@10 (commit to BM25 when p >= theta, theta tuned on the training
repos only), each with a repository-clustered bootstrap CI.

CPU-only. Run in the `llm` conda env.
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from embedding_premium import Router, has_identifiers  # noqa: E402
from lexical_rescue_taxonomy import PATTERNS  # noqa: E402
from run_retrieval import chunk_file, list_corpus, read_file, strip_pr_boilerplate  # noqa: E402

ROOT = _HERE.parent
SMOKE = DATA_ROOT / "smoke_test_results"
QRELS = {"cellA": DATA_ROOT / "release_src" / "qrels",
         "cellB": DATA_ROOT / "release_src" / "qrels_post_cutoff"}
CANON = {"cellA": SMOKE / "canon202607" / "jina05_cellA_c1000",
         "cellB": SMOKE / "canon202607" / "jina05_cellB_c1000"}
STRUCT_FEATURES = list(PATTERNS)  # 7 query-only features
GOLD_FEATURES_DROPPED = ["gold_basename", "rare_identifier"]
N_BOOT = 2000
RNG = np.random.default_rng(0)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_pool(pool_dir: Path) -> list[dict]:
    rows = []
    for repo_dir in sorted(p for p in pool_dir.iterdir() if p.is_dir()):
        for f in repo_dir.glob("smoke_test_*.jsonl"):
            for r in read_jsonl(f):
                if "dense_ndcg10" in r and "bm25_ndcg10" in r:
                    rows.append({"repo": repo_dir.name, "query_id": r["query_id"],
                                 "dense": float(r["dense_ndcg10"]),
                                 "bm25": float(r["bm25_ndcg10"])})
    return rows


def ceiling(rows: list[dict]) -> dict:
    d = np.array([r["dense"] for r in rows])
    b = np.array([r["bm25"] for r in rows])
    return {"n": len(rows), "dense": d.mean(), "bm25": b.mean(),
            "oracle": np.maximum(d, b).mean(),
            "headroom": np.maximum(d, b).mean() - d.mean(),
            "bm25_win_rate": float((b > d).mean()),
            "tie_rate": float((b == d).mean()),
            "both_zero_rate": float(((b == 0) & (d == 0)).mean())}


# ---------------------------------------------------------------- tau-hat
_ROUTERS: dict[str, Router] = {}


def repo_router(clones: Path, repo: str) -> Router:
    if repo not in _ROUTERS:
        chunks = []
        clone = clones / repo
        for rel in list_corpus(clone):
            text = read_file(clone, rel)
            if text.strip():
                chunks.extend(chunk_file(text, target_chars=1000, overlap_lines=10))
        _ROUTERS[repo] = Router.fit(chunks)
    return _ROUTERS[repo]


def features(query: str, tau_hat: float) -> dict:
    text = strip_pr_boilerplate(query)
    f = {name: float(bool(rx.search(text))) for name, rx in PATTERNS.items()}
    f["tau_hat"] = tau_hat
    f["has_identifiers"] = float(has_identifiers(text))
    f["log_chars"] = math.log1p(len(text))
    return f


# ---------------------------------------------------------------- D1
def loo_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray, make) -> np.ndarray:
    p = np.zeros(len(y))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if y[tr].min() == y[tr].max():
            p[te] = y[tr].mean()
            continue
        m = make().fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def loo_route(p: np.ndarray, dense: np.ndarray, bm25: np.ndarray,
              groups: np.ndarray, X, y, make) -> np.ndarray:
    """Routed nDCG with theta tuned on training repos' in-fold predictions
    (nested: inner LOO over the training repos), so the held-out repo never
    informs its own threshold."""
    out = dense.copy()
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        p_in = loo_predict(X[tr], y[tr], groups[tr], make)
        gain_by_theta = {t: ((bm25[tr] - dense[tr]) * (p_in >= t)).sum()
                         for t in np.unique(p_in)}
        theta, gain = max(gain_by_theta.items(), key=lambda kv: kv[1])
        if gain <= 0:
            continue  # never switch
        out[te] = np.where(p[te] >= theta, bm25[te], dense[te])
    return out


def clustered_ci(groups: np.ndarray, stat) -> list[float]:
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(N_BOOT):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        v = stat(idx)
        if v is not None and not math.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def auc_or_none(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clones", type=Path,
                    default=DATA_ROOT / "repos")
    ap.add_argument("--out", type=Path,
                    default=DATA_ROOT / "analysis" / "router_ceiling.json")
    args = ap.parse_args()
    report: dict = {"d0": {}, "loader_checks": {}, "d1": {}}

    # ---- D0 over every pool that has both retrievers
    pools = sorted(p for p in SMOKE.rglob("*") if p.is_dir()
                   and any(p.glob("*/smoke_test_*.jsonl")))
    for pool in pools:
        rows = load_pool(pool)
        if rows:
            report["d0"][str(pool.relative_to(SMOKE))] = ceiling(rows)

    # ---- canonical cells: tau-hat, features
    data = []
    for cell, pool in CANON.items():
        queries = {}
        for f in QRELS[cell].glob("*.jsonl"):
            for r in read_jsonl(f):
                queries[r["query_id"]] = r.get("problem_statement") or r.get("title", "")
        for r in load_pool(pool):
            q = queries[r["query_id"]]
            th = repo_router(args.clones, r["repo"]).tau_query(q)
            data.append({**r, "cell": cell, **features(q, th)})

    # loader check 1: tab:router Cell B quartiles (+0.276/+0.222/+0.168/+0.151)
    b = sorted((x for x in data if x["cell"] == "cellB"), key=lambda x: x["tau_hat"])
    qs = np.array_split(np.arange(len(b)), 4)
    report["loader_checks"]["cellB_tauhat_quartiles"] = [
        {"n": len(i), "range": [b[i[0]]["tau_hat"], b[i[-1]]["tau_hat"]],
         "mean_delta": float(np.mean([b[j]["dense"] - b[j]["bm25"] for j in i]))}
        for i in qs]
    # loader check 2: LOO hard switch on tau-hat (paper: A 0.532, B 0.545)
    for cell in CANON:
        rows = [x for x in data if x["cell"] == cell]
        repos = np.array([x["repo"] for x in rows])
        th = np.array([x["tau_hat"] for x in rows])
        d = np.array([x["dense"] for x in rows])
        bm = np.array([x["bm25"] for x in rows])
        routed = d.copy()
        for g in np.unique(repos):
            tr = repos != g
            best_t, best_gain = math.inf, 0.0
            for t in np.unique(th[tr]):
                gain = (bm[tr] - d[tr])[th[tr] >= t].sum()
                if gain > best_gain:
                    best_t, best_gain = t, gain
            te = repos == g
            routed[te] = np.where(th[te] >= best_t, bm[te], d[te])
        report["loader_checks"][f"{cell}_loo_hard_switch"] = {
            "dense": d.mean(), "routed": routed.mean()}

    # ---- D1: LOO-repo classifiers, cells pooled, repo = group
    feat_sets = {"structural7": STRUCT_FEATURES,
                 "structural7+tauhat+ids+len": STRUCT_FEATURES + ["tau_hat", "has_identifiers", "log_chars"]}
    models = {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0)),
        "gbt": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=2,
                                                  learning_rate=0.05, subsample=0.8, random_state=0),
    }
    groups = np.array([x["repo"] for x in data])
    cells = np.array([x["cell"] for x in data])
    y = np.array([float(x["bm25"] > x["dense"]) for x in data])
    dense = np.array([x["dense"] for x in data])
    bm25 = np.array([x["bm25"] for x in data])
    nontie = dense != bm25
    report["d1"]["n"] = int(len(y))
    report["d1"]["n_bm25_win"] = int(y.sum())
    report["d1"]["dropped_gold_features"] = GOLD_FEATURES_DROPPED
    for fs_name, cols in feat_sets.items():
        X = np.array([[x[c] for c in cols] for x in data])
        for m_name, make in models.items():
            p = loo_predict(X, y, groups, make)
            routed = loo_route(p, dense, bm25, groups, X, y, make)
            res = {}
            for scope, mask in [("pooled", np.ones(len(y), bool)),
                                ("cellA", cells == "cellA"), ("cellB", cells == "cellB")]:
                g = groups[mask]
                yy, pp, nt = y[mask], p[mask], nontie[mask]
                dd, rr = dense[mask], routed[mask]
                res[scope] = {
                    "auroc_all": auc_or_none(yy, pp),
                    "auroc_all_ci": clustered_ci(g, lambda i: auc_or_none(yy[i], pp[i])),
                    "auroc_nontie": auc_or_none(yy[nt], pp[nt]),
                    "auroc_nontie_ci": clustered_ci(
                        g, lambda i: auc_or_none(yy[i][nt[i]], pp[i][nt[i]])),
                    "dense": dd.mean(), "routed": rr.mean(),
                    "routed_minus_dense": (rr - dd).mean(),
                    "routed_minus_dense_ci": clustered_ci(g, lambda i: (rr[i] - dd[i]).mean()),
                    "frac_routed_to_bm25": float((rr != dd).mean()),
                }
            report["d1"][f"{fs_name}/{m_name}"] = res

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps({"loader_checks": report["loader_checks"],
                      "d1": report["d1"]}, indent=1, default=lambda v: round(float(v), 4)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
