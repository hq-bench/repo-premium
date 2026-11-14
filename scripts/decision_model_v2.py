#!/usr/bin/env python3
"""decision_model_v2.py — cost-aware decision model, DEVELOPMENT ONLY.

v1 (decision_model.py) chose among {dense, bm25@c1000, rrf} and failed its
preregistered held-out test (+0.003 [-0.012,+0.019]). Two findings since then
change the action set:
  * BM25 at c=1000 is a handicapped baseline: tuned BM25 matches dense on
    curated queries (chunk_rebase_ext.py);
  * generate-then-grep (BM25 on the query plus model-generated anchors) beats
    dense on curated queries and closes most of the raw premium.

Actions and per-query cost (one LLM call dominates an embedding, which
dominates BM25):
    bm25      BM25, tuned chunk size                        cost 0
    dense     jina-code-0.5b, c=1000                         cost 1 (embed)
    rrf_db    RRF(dense, bm25)                              cost 1
    gen       BM25 on query + generated anchors              cost 2 (LLM)
    rrf_dg    RRF(dense, gen)                               cost 3

Data: Cells A and B only (612 queries, 10 repositories) — the only pools
where every action exists. The scale-up repositories (seen once by v1's test)
have no generated anchors yet. The LOCKBOX (DeepSWE, R9 Rust/TS/Go) is never
loaded here.

Stage `headroom`: per-action means, oracle headroom per action subset, and
repository-block CIs. No model is fitted.
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from decision_model import ndcg10, rrf  # noqa: E402

ROOT = _HERE.parent
DATA = DATA_ROOT
CANON = DATA / "smoke_test_results/canon202607"
SWEEP = DATA / "smoke_test_results/bm25_sweep202610"
GENARM = DATA / "remote_pull/genarm/results"
GEN_SWEEP = DATA / "smoke_test_results/gen_sweep202610"
COST = {"bm25": 0, "dense": 1, "rrf_db": 1, "gen": 2, "rrf_dg": 3, "gen_t": 2, "rrf_dgt": 3}
N_BOOT = 10_000


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load(d: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for f in sorted(d.glob("*/smoke_test_django.jsonl")):
        for r in read_jsonl(f):
            out[(f.parent.name, r["query_id"])] = r
    return out


def tuned_c() -> str:  # noqa: D401
    """One chunk size for both cells (a deployed system cannot see the cell):
    the size with the best pooled BM25 mean over A and B."""
    best, best_v = None, -1.0
    for p in sorted(SWEEP.glob("bm25_cellA_c*")):
        c = p.name.split("_c")[-1]
        rows = list(load(p).values()) + list(load(SWEEP / f"bm25_cellB_c{c}").values())
        v = sum(r["bm25_ndcg10"] for r in rows) / len(rows)
        if v > best_v:
            best, best_v = c, v
    assert best is not None
    return best


def build(c: str) -> list[dict]:
    rows = []
    for cell in ("A", "B"):
        dense = load(CANON / f"jina05_cell{cell}_c1000")
        bm = load(SWEEP / f"bm25_cell{cell}_c{c}")
        gen = load(GENARM / f"expanded_cell{cell}")
        gen_t = load(GEN_SWEEP / f"gen_cell{cell}_c{c}") if (GEN_SWEEP / f"gen_cell{cell}_c{c}").is_dir() else {}
        for k in sorted(dense):
            if k not in bm or k not in gen:
                continue
            gold = dense[k]["gold_files"]
            d, b, g = dense[k]["dense_top10"], bm[k]["bm25_top10"], gen[k]["bm25_top10"]
            row = {"repo": k[0], "query_id": k[1], "cell": cell,
                   # django's A and B queries share a repository fold
                   "dense": ndcg10(d, gold), "bm25": ndcg10(b, gold),
                   "rrf_db": ndcg10(rrf(d, b), gold), "gen": ndcg10(g, gold),
                   "rrf_dg": ndcg10(rrf(d, g), gold),
                   "_lists": {"dense": d, "bm25": b, "gen": g}}
            assert abs(row["dense"] - dense[k]["dense_ndcg10"]) < 1e-9
            assert abs(row["bm25"] - bm[k]["bm25_ndcg10"]) < 1e-9
            assert abs(row["gen"] - gen[k]["bm25_ndcg10"]) < 1e-9
            if gen_t and k in gen_t:
                gt = gen_t[k]["bm25_top10"]
                row["gen_t"] = ndcg10(gt, gold)
                row["rrf_dgt"] = ndcg10(rrf(d, gt), gold)
                row["_lists"]["gen_t"] = gt
            rows.append(row)
    return rows


def boot_repo(rows, vec: np.ndarray, seed: int = 0) -> list[float]:
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["repo"]].append(i)
    groups = list(by.values())
    rng = np.random.default_rng(seed)
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = [i for g in rng.integers(0, len(groups), len(groups)) for i in groups[g]]
        vals[b] = vec[idx].mean()
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def headroom(rows) -> dict:
    acts = [a for a in COST if all(a in r for r in rows)]
    dense = np.array([r["dense"] for r in rows])
    out = {"n": len(rows), "n_repos": len({r["repo"] for r in rows}), "actions": acts,
           "always": {}, "oracle": {}}
    for a in acts:
        v = np.array([r[a] for r in rows])
        out["always"][a] = {"mean": float(v.mean()), "minus_dense": float((v - dense).mean()),
                            "ci": boot_repo(rows, v - dense),
                            "by_cell": {c: float(np.mean([r[a] for r in rows if r["cell"] == c]))
                                        for c in ("A", "B")}}
    subsets = [("dense", "bm25"), ("dense", "gen"), ("bm25", "gen"),
               ("dense", "bm25", "gen"), tuple(acts)]
    for s in subsets:
        if not all(a in acts for a in s):
            continue
        v = np.array([max(r[a] for a in s) for r in rows])
        # cheapest action that attains the oracle, for the oracle's mean cost
        cost = np.mean([min(COST[a] for a in s if r[a] == max(r[x] for x in s)) for r in rows])
        out["oracle"]["+".join(s)] = {"mean": float(v.mean()),
                                      "minus_dense": float((v - dense).mean()),
                                      "mean_cost": float(cost)}
    return out


# ------------------------------------------------------------ stage: route
# Base policy = rrf_db (embedding + tuned BM25, no LLM). The decision is
# whether to also pay for one LLM call and serve rrf_dgt instead. Every
# feature is available before that call: query-only (Q) and agreement
# between the dense and tuned-BM25 lists (P).
BASE, PAID = "rrf_db", "rrf_dgt"
BUDGETS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


def add_features(rows, clones: Path) -> list[str]:
    from decision_model import P_FEATS, Q_FEATS, p_features
    from router_ceiling import QRELS, features as q_features, repo_router
    text = {}
    for cell, key in (("A", "cellA"), ("B", "cellB")):
        for f in QRELS[key].glob("*.jsonl"):
            for q in read_jsonl(f):
                text[(cell, q["query_id"])] = q.get("problem_statement") or q.get("title", "")
    grep = {}
    for cell in ("A", "B"):
        for k, r in load(CANON / f"jina05_cell{cell}_c1000").items():
            grep[(cell,) + k] = r.get("grep_top10", [])
    for r in rows:
        q = text[(r["cell"], r["query_id"])]
        th = repo_router(clones, r["repo"]).tau_query(q)
        r.update(q_features(q, th))
        r.update(p_features(q, r["_lists"]["dense"], r["_lists"]["bm25"],
                            grep[(r["cell"], r["repo"], r["query_id"])]))
    return Q_FEATS + P_FEATS


def make_model(kind):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    return GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=0)


def loo_predict(rows, feats, kind) -> np.ndarray:
    X = np.array([[r[f] for f in feats] for r in rows], dtype=float)
    y = np.array([r[PAID] - r[BASE] for r in rows])
    repos = np.array([r["repo"] for r in rows])
    pred = np.zeros(len(rows))
    for g in np.unique(repos):
        tr, te = repos != g, repos == g
        pred[te] = make_model(kind).fit(X[tr], y[tr]).predict(X[te])
    return pred


def route(rows, clones: Path) -> dict:
    from sklearn.metrics import roc_auc_score
    feats_all = add_features(rows, clones)
    from decision_model import P_FEATS, Q_FEATS
    featsets = {"Q": Q_FEATS, "P": P_FEATS, "Q+P": feats_all}
    y = np.array([r[PAID] - r[BASE] for r in rows])
    base = np.array([r[BASE] for r in rows])
    dense = np.array([r["dense"] for r in rows])
    out = {"base": BASE, "paid": PAID, "n": len(rows),
           "gain_if_always_paid": float(y.mean()),
           "share_paid_better": float((y > 0).mean()), "share_paid_worse": float((y < 0).mean()),
           "share_tie": float((y == 0).mean()),
           "oracle_gain_over_base": float(np.maximum(y, 0).mean()),
           "oracle_llm_rate": float((y > 0).mean()), "models": {}}
    for fs, feats in featsets.items():
        for kind in ("ridge", "gbt"):
            pred = loo_predict(rows, feats, kind)
            order = np.argsort(-pred)
            curve = {}
            for f in BUDGETS:
                k = int(round(f * len(rows)))
                sel = np.zeros(len(rows), bool)
                sel[order[:k]] = True
                routed = np.where(sel, base + y, base)
                curve[str(f)] = {"routed": float(routed.mean()),
                                 "minus_base": float((routed - base).mean()),
                                 "random_minus_base": float(f * y.mean()),
                                 "share_of_paid_gain": float((routed - base).sum() / y.sum())
                                 if y.sum() else None}
            # model-vs-random at the 30% budget, repo-block CI on the difference
            k = int(round(0.3 * len(rows)))
            sel = np.zeros(len(rows), bool)
            sel[order[:k]] = True
            diff = np.where(sel, y, 0.0) - 0.3 * y
            out["models"][f"{kind}/{fs}"] = {
                "auroc_paid_better": float(roc_auc_score(y > 0, pred)) if 0 < (y > 0).sum() < len(y) else None,
                "curve": curve,
                "budget30_minus_random": float(diff.mean()),
                "budget30_minus_random_ci": boot_repo(rows, diff),
                "budget30_minus_dense_ci": boot_repo(rows, np.where(sel, base + y, base) - dense)}
    return out


# ------------------------------------------------------------ stage: qpp
# Post-retrieval score features (query performance prediction). BM25 scores
# come from qpp_scores.py --skip-dense; dense scores are added when the GPU
# re-encode exists. Two decisions are tested, each against its own target:
#   serve    which list to serve without an LLM: dense, bm25 or rrf_db
#            (target per action: its nDCG minus dense)
#   pay      whether to pay for generation: rrf_dgt minus rrf_db
QPP_DIRS = {"bm25": DATA / "smoke_test_results/qpp_bm25_202610",
            "dense": DATA / "smoke_test_results/qpp_scores202610"}
RERANK_DIR = DATA / "smoke_test_results/rerank_union202610"


def add_rerank(rows) -> bool:
    """jina-reranker-v3 over dense top-10 union BM25 top-10 (rerank_union.py)."""
    if not RERANK_DIR.is_dir():
        return False
    recs = {}
    for f in RERANK_DIR.glob("*.jsonl"):
        for r in read_jsonl(f):
            recs[(r["cell"], f.stem, r["query_id"])] = r
    for row in rows:
        r = recs[(row["cell"], row["repo"], row["query_id"])]
        row["rerank"] = ndcg10([p for p, _ in r["rerank"]], r["gold_files"])
    return True


def score_feats(prefix: str, top: list, mean: float, std: float) -> dict:
    s = np.array([x[1] for x in top], dtype=float)
    if len(s) < 10:  # corpora with < 10 files (a few lockbox snapshots); never on dev
        s = np.concatenate([s, np.full(10 - len(s), s[-1])])
    k10 = s[:10]
    eps = 1e-9
    z = (s - mean) / (std + eps)
    p = np.exp(k10 - k10.max())
    p /= p.sum()
    return {f"{prefix}_top1": s[0], f"{prefix}_gap12": s[0] - s[1],
            f"{prefix}_gap1_10": s[0] - s[9], f"{prefix}_ratio12": s[1] / (s[0] + eps),
            f"{prefix}_z1": z[0], f"{prefix}_z10": z[:10].mean(),
            # NQC: spread of the top-k relative to the corpus mean score
            f"{prefix}_nqc": k10.std() / (abs(mean) + eps),
            # WIG-style: mean top-k score above the corpus mean
            f"{prefix}_wig": (k10.mean() - mean) / (std + eps),
            f"{prefix}_ent10": float(-(p * np.log(p + eps)).sum())}


def add_qpp(rows) -> list[str]:
    feats = []
    for name, d in QPP_DIRS.items():
        if not d.is_dir():
            continue
        recs = {}
        for f in d.glob("*.jsonl"):
            for r in read_jsonl(f):
                if name in r:
                    recs[(r["cell"], f.stem, r["query_id"])] = r
        if not recs:
            continue
        for row in rows:
            r = recs[(row["cell"], row["repo"], row["query_id"])]
            row.update(score_feats(name, r[name], r[f"{name}_mean"], r[f"{name}_std"]))
        feats += [k for k in rows[0] if k.startswith(name + "_") and k not in ("bm25", "dense")]
    return feats


def eval_pay(rows, featsets) -> dict:
    from sklearn.metrics import roc_auc_score
    y = np.array([r[PAID] - r[BASE] for r in rows])
    base = np.array([r[BASE] for r in rows])
    out = {}
    for fs, feats in featsets.items():
        for kind in ("ridge", "gbt"):
            pred = loo_predict(rows, feats, kind)
            order = np.argsort(-pred)
            k = int(round(0.3 * len(rows)))
            sel = np.zeros(len(rows), bool)
            sel[order[:k]] = True
            diff = np.where(sel, y, 0.0) - 0.3 * y
            out[f"{kind}/{fs}"] = {"auroc": float(roc_auc_score(y > 0, pred)),
                                   "budget30_gain_over_base": float(np.where(sel, y, 0).mean()),
                                   "budget30_minus_random": float(diff.mean()),
                                   "ci": boot_repo(rows, diff)}
    return out


def eval_serve(rows, featsets) -> dict:
    """Pick dense / bm25 / rrf_db per query; inner-LOO margin as in v1."""
    from sklearn.metrics import roc_auc_score
    acts = ["dense", "bm25", "rrf_db"]
    dense = np.array([r["dense"] for r in rows])
    best_static = max(acts, key=lambda a: np.mean([r[a] for r in rows]))
    static = np.array([r[best_static] for r in rows])
    repos = np.array([r["repo"] for r in rows])
    out = {"best_static": best_static,
           "oracle_minus_static": float(np.mean([max(r[a] for a in acts) for r in rows]) - static.mean())}
    for fs, feats in featsets.items():
        for kind in ("ridge", "gbt"):
            X = np.array([[r[f] for f in feats] for r in rows], dtype=float)
            Y = np.array([[r[a] - r[best_static] for a in acts] for r in rows])
            P = np.zeros_like(Y)
            for g in np.unique(repos):
                tr, te = repos != g, repos == g
                for j in range(len(acts)):
                    P[te, j] = make_model(kind).fit(X[tr], Y[tr, j]).predict(X[te])
            # serve the predicted-best action only if it beats the static one by > 0
            j = P.argmax(1)
            routed = np.array([r[acts[jj]] if P[i, jj] > 0 else r[best_static]
                               for i, (r, jj) in enumerate(zip(rows, j))])
            nonstatic = np.array([max(r[a] for a in acts) > r[best_static] for r in rows])
            out[f"{kind}/{fs}"] = {
                "routed_minus_static": float((routed - static).mean()),
                "ci": boot_repo(rows, routed - static),
                "routed_minus_dense": float((routed - dense).mean()),
                "deviation_rate": float(np.mean([P[i, jj] > 0 for i, jj in enumerate(j)])),
                "auroc_nonstatic_better": float(roc_auc_score(nonstatic, P.max(1)))}
    return out


def qpp(rows, clones: Path) -> dict:
    from decision_model import P_FEATS, Q_FEATS
    add_features(rows, clones)
    s_feats = add_qpp(rows)
    featsets = {"S": s_feats, "Q+P": Q_FEATS + P_FEATS, "Q+P+S": Q_FEATS + P_FEATS + s_feats}
    out = {"score_features": s_feats, "serve": eval_serve(rows, featsets),
           "pay": eval_pay(rows, featsets)}
    if add_rerank(rows):
        dense = np.array([r["dense"] for r in rows])
        rr = np.array([r["rerank"] for r in rows])
        st = np.array([r[out["serve"]["best_static"]] for r in rows])
        out["rerank_baseline"] = {
            "mean": float(rr.mean()),
            "minus_dense": float((rr - dense).mean()), "minus_dense_ci": boot_repo(rows, rr - dense),
            "minus_static": float((rr - st).mean()), "minus_static_ci": boot_repo(rows, rr - st),
            "by_cell": {c: float(np.mean([r["rerank"] for r in rows if r["cell"] == c])) for c in ("A", "B")}}
    return out


# ------------------------------------------------------------ stage: freeze
FROZEN = DATA / "analysis/decision_model_v2_artifacts"
PREREG = """PREREGISTRATION — decision model v2, serve decision (frozen 2026-10-05,
before any lockbox query was scored).

Model. Gradient-boosted regressors (n_estimators=150, max_depth=2,
learning_rate=0.05, subsample=0.8, random_state=0), one per action a in
{dense, bm25, rrf_db}, each predicting nDCG@10(a) - nDCG@10(rrf_db) from the
feature set Q+P+S (query-only, dense/BM25 list agreement, dense and BM25
retrieval-score features), fitted once on all 612 development queries
(Cells A and B). Rule: serve argmax_a of the predicted gain if it is > 0,
else rrf_db. Retrievers: dense = jina-code-0.5b at c=1000 max-pooled per
file; bm25 = BM25 over whole files; rrf_db = RRF (k=60) of their top-10 lists.

Selection. Chosen on dev as the only one of six serve models whose
leave-one-repository-out gain over always-rrf_db excluded zero (+0.013,
95% CI [+0.005,+0.036]). About 30 model/feature combinations were tried
across v1 and v2; this is the single candidate carried to the lockbox.

Lockbox. DeepSWE (main eligible tasks) and the R9 pools (TypeScript,
Go, Rust), run with each pool's own canonical runner settings. Neither
was scored for any v2 quantity before this freeze.

Primary. Mean of routed minus always-rrf_db nDCG@10 over the pooled lockbox
queries; repository-clustered bootstrap (10,000 resamples, seed 0).
SUCCESS iff the 95% CI lower bound > 0.
Secondary (no multiplicity claim): the same per lockbox family; always-rrf_db
minus always-dense with non-inferiority margin -0.05 (budget-table check);
oracle headroom. The generation tier is not tested (no lockbox anchors).
"""


def freeze(rows, clones: Path) -> dict:
    import hashlib
    import pickle
    from decision_model import P_FEATS, Q_FEATS
    add_features(rows, clones)
    feats = Q_FEATS + P_FEATS + add_qpp(rows)
    acts = ["dense", "bm25", "rrf_db"]
    X = np.array([[r[f] for f in feats] for r in rows], dtype=float)
    models = {a: make_model("gbt").fit(X, np.array([r[a] - r["rrf_db"] for r in rows]))
              for a in acts}
    FROZEN.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps({"features": feats, "actions": acts, "models": models})
    (FROZEN / "serve_gbt_QPS.pkl").write_bytes(blob)
    spec = {"features": feats, "actions": acts, "base": "rrf_db", "rule": "argmax gain > 0 else base",
            "sha256": hashlib.sha256(blob).hexdigest(), "n_dev": len(rows),
            "preregistration": PREREG}
    (FROZEN / "serve_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    (FROZEN / "PREREGISTRATION.txt").write_text(PREREG)
    return spec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["headroom", "route", "qpp", "freeze"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--clones", type=Path, default=DATA_ROOT / "repos")
    args = ap.parse_args()
    c = tuned_c()
    rows = build(c)
    if args.stage == "freeze":
        spec = freeze(rows, args.clones)
        print("frozen", spec["sha256"], len(spec["features"]), "features")
        return 0
    if args.stage == "qpp":
        res = {"tuned_c": c, **qpp(rows, args.clones)}
        (args.out or DATA / "analysis/decision_model_v2_qpp.json").write_text(
            json.dumps(res, indent=2) + "\n")
        print("score features:", res["score_features"])
        print("SERVE: best static =", res["serve"]["best_static"],
              "oracle-static", round(res["serve"]["oracle_minus_static"], 4))
        for m, v in res["serve"].items():
            if isinstance(v, dict):
                print(f"  {m:10s} routed-static {v['routed_minus_static']:+.4f} "
                      f"{[round(x, 4) for x in v['ci']]} dev-rate {v['deviation_rate']:.2f} "
                      f"auroc {v['auroc_nonstatic_better']:.3f}")
        if "rerank_baseline" in res:
            print("RERANK baseline:", json.dumps(res["rerank_baseline"]))
        print("PAY (rrf_dgt vs rrf_db, 30% LLM budget):")
        for m, v in res["pay"].items():
            print(f"  {m:10s} auroc {v['auroc']:.3f} gain {v['budget30_gain_over_base']:+.4f} "
                  f"model-random {v['budget30_minus_random']:+.4f} {[round(x, 4) for x in v['ci']]}")
        return 0
    if args.stage == "route":
        res = {"tuned_c": c, **route(rows, args.clones)}
        (args.out or DATA / "analysis/decision_model_v2_route.json").write_text(
            json.dumps(res, indent=2) + "\n")
        print({k: v for k, v in res.items() if k != "models"})
        for m, v in res["models"].items():
            c30 = v["curve"]["0.3"]
            print(f"  {m:10s} auroc={v['auroc_paid_better']:.3f}  @30%: {c30['minus_base']:+.4f} "
                  f"(random {c30['random_minus_base']:+.4f}, {c30['share_of_paid_gain']:.0%} of paid gain) "
                  f"model-random {v['budget30_minus_random']:+.4f} "
                  f"{[round(x, 4) for x in v['budget30_minus_random_ci']]}")
        return 0
    res = {"tuned_c": c, **headroom(rows)}
    (args.out or DATA / "analysis/decision_model_v2_headroom.json").write_text(
        json.dumps(res, indent=2) + "\n")
    print(f"tuned c={c}  n={res['n']}  repos={res['n_repos']}")
    for a, v in res["always"].items():
        print(f"  always {a:8s} {v['mean']:.4f}  vs dense {v['minus_dense']:+.4f} "
              f"{[round(x, 4) for x in v['ci']]}  A={v['by_cell']['A']:.3f} B={v['by_cell']['B']:.3f}")
    for s, v in res["oracle"].items():
        print(f"  oracle {s:40s} {v['mean']:.4f} vs dense {v['minus_dense']:+.4f} cost {v['mean_cost']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
