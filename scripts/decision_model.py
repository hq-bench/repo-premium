#!/usr/bin/env python3
"""decision_model.py — a light per-query decision model for code search.

PREREGISTRATION (written 2026-10-03, before any held-out number was computed)
------------------------------------------------------------------------------
Question. Can a light decision model choose, per query, how to search well
enough to beat always-dense on repositories it has never seen?

Action set A = {dense, bm25, rrf(dense, bm25)} (RRF k=60 over the stored
top-10 file lists). Generated-anchor BM25 and rrf(dense, genarm) are NOT in
A: their rankings exist for the development cells only, not for the test
repositories, and A must be identical on dev and test.

Encoder. Canonical jina-code-0.5b (c=1000) on both dev and test.

Development. Canonical Cell A + Cell B (612 queries, 10 repositories),
leave-one-repository-out (django's A and B queries share one fold). Every
threshold is tuned by an inner leave-one-repository-out over the training
repositories only.

Test. The matched scale-up, curated and raw arms, minus the seven
repositories shared with the 19-repository law battery (microsoft/TypeScript,
pandas-dev/pandas, pydantic/pydantic, python-pillow/Pillow,
python-poetry/poetry, tokio-rs/tokio, tornadoweb/tornado): 29 repositories.
The model is fitted once on all of dev, frozen, pickled, and applied without
refitting.

LOCKBOX. DeepSWE and the R9 multilingual pools (Rust/TS/Go) are not loaded,
scored or inspected by this script. The frozen artifacts are saved so they
can be scored once later.

Primary model. Cost-sensitive gradient-boosted regression ("reg_gbt") on
query-only + post-retrieval features (feature set "Q+P"): predicts each
action's nDCG@10 gain over dense and takes the argmax if its predicted gain
exceeds a margin theta (tuned on dev), else dense. All other model/feature
combinations are secondary and reported without multiplicity claims.

Primary metric. Routed nDCG@10 minus always-dense on the test set.
Inference: repository-then-task clustered bootstrap (a task's curated and raw
queries move together), 10,000 resamples, seed 0.
SUCCESS iff the 95% CI lower bound of (routed - always-dense) on test > 0.
Also reported: routed - (best single action chosen on dev), and the oracle.

Features. Q (query-only): seven structural patterns, has_identifiers,
log_chars, tau-hat (top-5 mean IDF of the stripped query against the
repository's chunk IDF). P (post-retrieval, list-only): the caches store
ranked file lists but NO retrieval scores, so score-based signals (top-1
score, score gaps, score entropy) are unavailable; P uses list agreement
between dense, BM25 and grep top-10s (overlap@1/3/5/10, rank-biased overlap
p=0.9, cross-ranks of each top-1), test-file share, directory dispersion,
union size, and whether the query names a file stem present in a list.
------------------------------------------------------------------------------
CPU only; run in the `llm` conda env.
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from router_ceiling import CANON, QRELS, read_jsonl, repo_router, features as q_features  # noqa: E402
from lexical_rescue_taxonomy import PATTERNS  # noqa: E402

ROOT = _HERE.parent
DATA = DATA_ROOT
SCALEUP = DATA / "scaleup_pull"
ANALYSIS = DATA / "analysis"
ART = ANALYSIS / "decision_model_artifacts"
OVERLAP_REPOS = {"microsoft/TypeScript", "pandas-dev/pandas", "pydantic/pydantic",
                 "python-pillow/Pillow", "python-poetry/poetry", "tokio-rs/tokio",
                 "tornadoweb/tornado"}
ACTIONS = ["dense", "bm25", "rrf"]
RRF_K = 60
N_BOOT = 10_000
Q_FEATS = list(PATTERNS) + ["has_identifiers", "log_chars", "tau_hat"]
P_FEATS = ["ov1", "ov3", "ov5", "ov10", "rbo", "rank_d1_in_b", "rank_b1_in_d",
           "ov10_grep_bm25", "ov10_grep_dense", "rank_g1_in_d", "rank_g1_in_b",
           "test_share_d", "test_share_b", "dirs_d", "dirs_b", "union",
           "qstem_d", "qstem_b", "qstem_d1", "qstem_b1"]
FEATSETS = {"Q": Q_FEATS, "P": P_FEATS, "Q+P": Q_FEATS + P_FEATS}


# ---------------------------------------------------------------- metrics
def ndcg10(lst: list[str], gold: list[str]) -> float:
    g = set(gold)
    dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(lst[:10]) if x in g)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(g), 10)))
    return dcg / idcg if idcg else 0.0


def rrf(a: list[str], b: list[str], k: int = RRF_K) -> list[str]:
    score: dict[str, float] = {}
    first: dict[str, tuple[int, int]] = {}
    for li, lst in enumerate((a, b)):
        for r, x in enumerate(lst[:10]):
            score[x] = score.get(x, 0.0) + 1.0 / (k + r + 1)
            first.setdefault(x, (r, li))
    return sorted(score, key=lambda x: (-score[x], first[x]))[:10]


# ---------------------------------------------------------------- features
_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def rbo(a: list[str], b: list[str], p: float = 0.9) -> float:
    s, sa, sb = 0.0, set(), set()
    for d in range(1, 11):
        if d <= len(a):
            sa.add(a[d - 1])
        if d <= len(b):
            sb.add(b[d - 1])
        s += p ** (d - 1) * len(sa & sb) / d
    return (1 - p) * s / (1 - p ** 10)


def ov(a, b, k):
    return len(set(a[:k]) & set(b[:k])) / k


def xrank(x, lst):
    return lst.index(x) + 1 if x in lst else 11


def stem(path: str) -> str:
    return Path(path).stem.lower()


def p_features(query: str, d: list[str], b: list[str], g: list[str]) -> dict:
    qt = {t.lower() for t in _TOK.findall(query)}
    f = {"ov1": float(bool(d) and bool(b) and d[0] == b[0]),
         "ov3": ov(d, b, 3), "ov5": ov(d, b, 5), "ov10": ov(d, b, 10),
         "rbo": rbo(d, b),
         "rank_d1_in_b": xrank(d[0], b) if d else 11,
         "rank_b1_in_d": xrank(b[0], d) if b else 11,
         "ov10_grep_bm25": ov(g, b, 10), "ov10_grep_dense": ov(g, d, 10),
         "rank_g1_in_d": xrank(g[0], d) if g else 11,
         "rank_g1_in_b": xrank(g[0], b) if g else 11,
         "test_share_d": np.mean(["test" in x.lower() for x in d]) if d else 0.0,
         "test_share_b": np.mean(["test" in x.lower() for x in b]) if b else 0.0,
         "dirs_d": len({str(Path(x).parent) for x in d}) / 10,
         "dirs_b": len({str(Path(x).parent) for x in b}) / 10,
         "union": len(set(d) | set(b)) / 20,
         "qstem_d": float(any(stem(x) in qt for x in d)),
         "qstem_b": float(any(stem(x) in qt for x in b)),
         "qstem_d1": float(bool(d) and stem(d[0]) in qt),
         "qstem_b1": float(bool(b) and stem(b[0]) in qt)}
    return {k: float(v) for k, v in f.items()}


def make_row(base: dict, query: str, tau_hat: float, r: dict) -> dict:
    d, b, g = r["dense_top10"], r["bm25_top10"], r.get("grep_top10", [])
    gold = r["gold_files"]
    fused = rrf(d, b)
    row = {**base, "dense": ndcg10(d, gold), "bm25": ndcg10(b, gold),
           "rrf": ndcg10(fused, gold)}
    assert abs(row["dense"] - r["dense_ndcg10"]) < 1e-9
    assert abs(row["bm25"] - r["bm25_ndcg10"]) < 1e-9
    row.update(q_features(query, tau_hat))
    row.update(p_features(query, d, b, g))
    return row


# ---------------------------------------------------------------- data
def load_dev(clones: Path) -> list[dict]:
    rows = []
    for cell, pool in CANON.items():
        queries = {}
        for f in QRELS[cell].glob("*.jsonl"):
            for q in read_jsonl(f):
                queries[q["query_id"]] = q.get("problem_statement") or q.get("title", "")
        for repo_dir in sorted(p for p in pool.iterdir() if p.is_dir()):
            for r in read_jsonl(repo_dir / "smoke_test_django.jsonl"):
                q = queries[r["query_id"]]
                th = repo_router(clones, repo_dir.name).tau_query(q)
                rows.append(make_row({"query_id": r["query_id"], "repo": repo_dir.name,
                                      "task": r["query_id"], "cell": cell, "lang": "python"},
                                     q, th, r))
    return rows


def load_test(tauhat_file: Path) -> list[dict]:
    manifest = {m["task_id"]: m for m in read_jsonl(SCALEUP / "splits_v2" / "retrieval_tasks_v2.jsonl")}
    tau = {t["query_id"]: t["tau_hat"] for t in read_jsonl(tauhat_file)}
    rows = []
    retr = SCALEUP / "experiment_v1" / "retrieval" / "jina-0.5b"
    for task_id, m in sorted(manifest.items()):
        if m["repo"] in OVERLAP_REPOS:
            continue
        f = retr / task_id / "smoke_test_django.jsonl"
        if not f.exists():
            continue
        for r in read_jsonl(f):
            arm = r["query_id"].split("::")[1]
            q = m["curated_query"] if arm == "curated" else m["raw_query"]
            rows.append(make_row({"query_id": r["query_id"], "repo": m["repo"],
                                  "task": task_id, "cell": arm,
                                  "lang": m["language_group"]},
                                 q, tau[r["query_id"]], r))
    return rows


# ---------------------------------------------------------------- models
def best_action(row) -> int:
    vals = [row[a] for a in ACTIONS]
    best = max(vals)
    for i in (0, 2, 1):  # ties: dense, then rrf, then bm25
        if vals[i] == best:
            return i
    raise AssertionError


class Decider:
    """Fits on rows; scores() returns per-action predicted gain over dense
    (index 0 is always 0); decide() applies margin theta."""

    def __init__(self, kind: str, feats: list[str]):
        self.kind, self.feats, self.theta = kind, feats, math.inf

    def _X(self, rows):
        return np.array([[r[c] for c in self.feats] for r in rows], dtype=float)

    def fit(self, rows):
        X = self._X(rows)
        if self.kind in ("clf_logreg", "clf_gbt"):
            y = np.array([best_action(r) for r in rows])
            self.classes_ = np.unique(y)
            if len(self.classes_) == 1:
                self.m = None
                return self
            self.m = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
                      if self.kind == "clf_logreg" else
                      GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                                 learning_rate=0.05, subsample=0.8,
                                                 random_state=0)).fit(X, y)
        else:
            self.m = []
            for a in ACTIONS[1:]:
                y = np.array([r[a] - r["dense"] for r in rows])
                reg = (GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                                 learning_rate=0.05, subsample=0.8,
                                                 random_state=0)
                       if self.kind == "reg_gbt" else
                       make_pipeline(StandardScaler(), Ridge(alpha=10.0)))
                self.m.append(reg.fit(X, y))
        return self

    def scores(self, rows) -> np.ndarray:
        X = self._X(rows)
        out = np.zeros((len(rows), len(ACTIONS)))
        if self.kind in ("clf_logreg", "clf_gbt"):
            if self.m is None:
                return out
            P = np.zeros((len(rows), len(ACTIONS)))
            P[:, self.classes_] = self.m.predict_proba(X)
            out[:, 1:] = P[:, 1:] - P[:, [0]]
        else:
            for j, reg in enumerate(self.m, start=1):
                out[:, j] = reg.predict(X)
        return out

    @staticmethod
    def choose(S: np.ndarray, theta: float) -> np.ndarray:
        j = S[:, 1:].argmax(1) + 1
        return np.where(S[np.arange(len(S)), j] > theta, j, 0)


def realized(rows, choice) -> np.ndarray:
    return np.array([r[ACTIONS[c]] for r, c in zip(rows, choice)])


def tune_theta(rows, kind, feats) -> float:
    """Inner LOO-repo over `rows`; theta maximizing pooled realized gain
    (inf = never deviate from dense)."""
    repos = np.array([r["repo"] for r in rows])
    S = np.zeros((len(rows), len(ACTIONS)))
    for g in np.unique(repos):
        tr = [r for r, x in zip(rows, repos) if x != g]
        te_idx = np.flatnonzero(repos == g)
        S[te_idx] = Decider(kind, feats).fit(tr).scores([rows[i] for i in te_idx])
    dense = np.array([r["dense"] for r in rows])
    best_t, best_gain = math.inf, 0.0
    cand = np.unique(S[:, 1:].max(1))
    for t in np.concatenate([cand - 1e-12]):
        gain = (realized(rows, Decider.choose(S, t)) - dense).sum()
        if gain > best_gain + 1e-12:
            best_t, best_gain = t, gain
    return best_t


def dev_loo(rows, kind, feats):
    repos = np.array([r["repo"] for r in rows])
    choice = np.zeros(len(rows), dtype=int)
    S_all = np.zeros((len(rows), len(ACTIONS)))
    for g in np.unique(repos):
        tr = [r for r, x in zip(rows, repos) if x != g]
        te_idx = np.flatnonzero(repos == g)
        theta = tune_theta(tr, kind, feats)
        m = Decider(kind, feats).fit(tr)
        S = m.scores([rows[i] for i in te_idx])
        S_all[te_idx] = S
        choice[te_idx] = Decider.choose(S, theta)
    return choice, S_all


# ---------------------------------------------------------------- inference
def boot_repo_task(rows, stat_vec: np.ndarray, rng) -> list[float]:
    """Repository-then-task clustered bootstrap of mean(stat_vec)."""
    by_repo: dict[str, dict[str, list[int]]] = {}
    for i, r in enumerate(rows):
        by_repo.setdefault(r["repo"], {}).setdefault(r["task"], []).append(i)
    repos = list(by_repo)
    tasks = {g: list(v.values()) for g, v in by_repo.items()}
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = []
        for g in rng.choice(len(repos), size=len(repos), replace=True):
            tl = tasks[repos[g]]
            for t in rng.choice(len(tl), size=len(tl), replace=True):
                idx.extend(tl[t])
        vals[b] = stat_vec[idx].mean()
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarize(rows, choice, S, rng, best_single: str | None = None) -> dict:
    dense = np.array([r["dense"] for r in rows])
    real = realized(rows, choice)
    oracle = np.array([max(r[a] for a in ACTIONS) for r in rows])
    gain = real - dense
    out = {"n": len(rows), "n_repos": len({r["repo"] for r in rows}),
           "always": {a: float(np.mean([r[a] for r in rows])) for a in ACTIONS},
           "oracle": float(oracle.mean()), "routed": float(real.mean()),
           "routed_minus_dense": float(gain.mean()),
           "routed_minus_dense_ci": boot_repo_task(rows, gain, rng),
           "oracle_minus_dense": float((oracle - dense).mean()),
           "action_dist": {a: float((choice == i).mean()) for i, a in enumerate(ACTIONS)}}
    if best_single:
        bs = np.array([r[best_single] for r in rows])
        out["best_single_on_dev"] = best_single
        out["routed_minus_best_single"] = float((real - bs).mean())
        out["routed_minus_best_single_ci"] = boot_repo_task(rows, real - bs, rng)
    # AUROC: is some non-dense action strictly better than dense?
    y = np.array([max(r["bm25"], r["rrf"]) > r["dense"] for r in rows])
    sc = S[:, 1:].max(1)
    nontie = np.array([not (r["bm25"] == r["dense"] == r["rrf"]) for r in rows])
    out["auroc_nondense_better"] = (float(roc_auc_score(y[nontie], sc[nontie]))
                                    if 0 < y[nontie].sum() < nontie.sum() else None)
    orc = np.array([best_action(r) for r in rows])
    out["accuracy_vs_oracle_action"] = float((orc == choice).mean())
    out["always_dense_accuracy_vs_oracle"] = float((orc == 0).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clones", type=Path, default=DATA_ROOT / "repos")
    ap.add_argument("--tauhat", type=Path, default=ANALYSIS / "decision_model_tauhat_scaleup.jsonl")
    ap.add_argument("--stage", choices=["dev", "test"], default="dev")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    dev = load_dev(args.clones)
    dev_means = {a: float(np.mean([r[a] for r in dev])) for a in ACTIONS}
    best_single = max(dev_means, key=dev_means.get)
    report = {"preregistered_primary": "reg_gbt/Q+P", "actions": ACTIONS,
              "dev_means": dev_means, "best_single_on_dev": best_single,
              "unavailable_features": ["top-1 score", "top1-top2 score gap",
                                       "score std/entropy (caches store no scores)"],
              "dev": {}, "test": {}}
    kinds = ["reg_gbt", "reg_ridge", "clf_logreg", "clf_gbt"]
    for fs, feats in FEATSETS.items():
        for kind in kinds:
            choice, S = dev_loo(dev, kind, feats)
            report["dev"][f"{kind}/{fs}"] = summarize(dev, choice, S, rng)
            print(kind, fs, round(report["dev"][f"{kind}/{fs}"]["routed_minus_dense"], 4),
                  report["dev"][f"{kind}/{fs}"]["routed_minus_dense_ci"], flush=True)
    # importances of the primary model fitted on all dev
    prim = Decider("reg_gbt", FEATSETS["Q+P"]).fit(dev)
    imp = np.mean([m.feature_importances_ for m in prim.m], axis=0)
    report["primary_importances"] = dict(sorted(zip(prim.feats, map(float, imp)),
                                                key=lambda kv: -kv[1]))
    if args.stage == "test":
        ART.mkdir(parents=True, exist_ok=True)
        test = load_test(args.tauhat)
        for fs, feats in FEATSETS.items():
            for kind in kinds:
                theta = tune_theta(dev, kind, feats)
                m = Decider(kind, feats).fit(dev)
                m.theta = theta
                with open(ART / f"{kind}__{fs.replace('+', 'plus')}.pkl", "wb") as fo:
                    pickle.dump(m, fo)
                S = m.scores(test)
                choice = Decider.choose(S, theta)
                res = summarize(test, choice, S, rng, best_single)
                res["theta"] = None if math.isinf(theta) else float(theta)
                for scope in ("curated", "raw"):
                    idx = [i for i, r in enumerate(test) if r["cell"] == scope]
                    sub = [test[i] for i in idx]
                    res[scope] = {"routed_minus_dense": float(
                        (realized(sub, choice[idx]) - np.array([r["dense"] for r in sub])).mean()),
                        "oracle_minus_dense": float(np.mean(
                            [max(r[a] for a in ACTIONS) - r["dense"] for r in sub]))}
                langs = sorted({r["lang"] for r in test})
                res["by_lang"] = {}
                for lg in langs:
                    idx = [i for i, r in enumerate(test) if r["lang"] == lg]
                    sub = [test[i] for i in idx]
                    res["by_lang"][lg] = {"n": len(sub), "routed_minus_dense": float(
                        (realized(sub, choice[idx]) - np.array([r["dense"] for r in sub])).mean())}
                report["test"][f"{kind}/{fs}"] = res
                print("TEST", kind, fs, round(res["routed_minus_dense"], 4),
                      res["routed_minus_dense_ci"], flush=True)
        prim_res = report["test"]["reg_gbt/Q+P"]
        report["verdict"] = {"primary": "reg_gbt/Q+P",
                             "test_gain": prim_res["routed_minus_dense"],
                             "test_ci": prim_res["routed_minus_dense_ci"],
                             "success": prim_res["routed_minus_dense_ci"][0] > 0}
        (ART / "config.json").write_text(json.dumps({
            "actions": ACTIONS, "rrf_k": RRF_K, "featsets": FEATSETS,
            "encoder": "jina-code-0.5b", "chunk_chars": 1000,
            "trained_on": "canonical Cell A + Cell B (612 queries)",
            "primary": "reg_gbt__QplusP.pkl",
            "feature_code": "scripts/decision_model.py (make_row, p_features) + "
                            "router_ceiling.features + embedding_premium.Router.tau_query",
            "lockbox": "DeepSWE, R9 Rust/TS/Go — not yet scored"}, indent=2))
    out = ANALYSIS / f"decision_model_{args.stage}.json"
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
