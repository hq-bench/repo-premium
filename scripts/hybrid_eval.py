"""hybrid_eval.py — Leave-one-repo-out α-blend RRF vs. baselines.

For the §6 R-ROUTER evaluation requested by the reviewer pass: holds out
one repo at a time, calibrates the τ-router thresholds on the other
nine, then evaluates per-query α-blend RRF on the held-out repo against
four baselines:

  - dense alone (current SOTA on \\benchmark)
  - BM25 alone
  - fixed-α RRF (α = 0.3, 0.5; standard hybrid baselines)
  - router α-blend RRF (α = router.alpha_blend(q))

Hybrid score per file is rank-based RRF a la Cormack et al. 2009:
  score(d) = α * 1/(k + rank_BM25(d)) + (1 - α) * 1/(k + rank_dense(d))
where rank_X(d) is +∞ if d does not appear in retriever X's top-N list,
in which case the corresponding term is 0. We use k = 60 (the standard
RRF smoothing constant) and N = 10 (the depth available in the smoke
test JSONLs).

Note on rank depth: with top-10 only, the hybrid can only re-order
files that appear in at least one retriever's top-10. This is a
lower bound on the true hybrid lift (a top-50 sweep could lift more
golds into top-10), and we report it as such.

Outputs JSON to --out with per-query and per-repo nDCG@10 deltas.

CPU-only; no torch dependency.
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from embedding_premium import Router  # noqa: E402
from run_retrieval import (  # noqa: E402
    chunk_file,
    list_corpus,
    read_file,
)

RRF_K = 60
TOP_N = 10  # depth available in smoke_test JSONLs

REPOS = [
    "astropy__astropy",
    "django__django",
    "matplotlib__matplotlib",
    "psf__requests",
    "pydata__xarray",
    "pylint-dev__pylint",
    "pytest-dev__pytest",
    "scikit-learn__scikit-learn",
    "sphinx-doc__sphinx",
    "sympy__sympy",
]


_CHUNK_CACHE: dict[str, list[str]] = {}


def repo_chunks(clones_dir: Path, repo: str) -> list[str]:
    """Chunk one repo's corpus once and cache (folds reuse the chunks)."""
    if repo not in _CHUNK_CACHE:
        chunks: list[str] = []
        clone_dir = clones_dir / repo
        if clone_dir.is_dir():
            for rel in list_corpus(clone_dir):
                text = read_file(clone_dir, rel)
                if not text.strip():
                    continue
                chunks.extend(chunk_file(text, target_chars=1000, overlap_lines=10))
        _CHUNK_CACHE[repo] = chunks
    return _CHUNK_CACHE[repo]


def build_router_for_repos(clones_dir: Path, repos: Iterable[str]) -> Router:
    """Build a Router whose IDF table is fit on the union of the listed
    repos' chunked corpora. Used for leave-one-repo-out calibration."""
    chunks: list[str] = []
    for repo in repos:
        chunks.extend(repo_chunks(clones_dir, repo))
    return Router.fit(chunks)


def calibrate_router(
    router: Router,
    smoke_root: Path,
    qrel_dir: Path,
    repos: Iterable[str],
    smoke_subdir: str,
) -> tuple[Router, list[tuple[float, float]]]:
    """Re-fit (semantic, hybrid) thresholds from quartile boundaries of
    the calibration repos' per-query (mean_idf, Δ) pairs. Also returns
    the raw pairs so callers can calibrate additional rules (e.g. the
    hard-switch threshold)."""
    pairs: list[tuple[float, float]] = []
    for repo in repos:
        smoke_dir = smoke_root / smoke_subdir / repo
        if not smoke_dir.is_dir():
            continue
        smoke_files = list(smoke_dir.glob("smoke_test_*.jsonl"))
        if not smoke_files:
            continue
        qrel_path = qrel_dir / f"{repo}.jsonl"
        if not qrel_path.exists():
            continue
        queries = load_queries(qrel_path)
        for line in smoke_files[0].read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            qid = r["query_id"]
            if qid not in queries:
                continue
            m = router.tau_query(queries[qid])
            d = float(r["delta_ndcg10"])
            pairs.append((m, d))
    if pairs:
        router.calibrate(pairs)
    return router, pairs


def calibrate_hard_threshold(pairs: list[tuple[float, float]]) -> float:
    """LOO-honest hard-switch threshold: commit to BM25 outright on
    queries with tau-hat >= theta. Choose theta maximizing the pooled
    nDCG gain on the calibration pairs, i.e. sum of (-delta) over the
    tail {tau-hat >= theta}; if no tail has positive gain (mean delta
    stays positive everywhere), return +inf ("never switch")."""
    if not pairs:
        return float("inf")
    ms = sorted({m for m, _ in pairs})
    best_theta, best_gain = float("inf"), 0.0
    for theta in ms:
        gain = sum(-d for m, d in pairs if m >= theta)
        if gain > best_gain:
            best_theta, best_gain = theta, gain
    return best_theta


def load_queries(qrel_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in qrel_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["query_id"]] = r.get("problem_statement", "") or r.get("title", "")
    return out


def rrf_fuse(
    bm25_top: list[str],
    dense_top: list[str],
    alpha: float,
    k: int = RRF_K,
) -> list[str]:
    """Return α-blended RRF fused ranking. alpha is the weight on BM25."""
    scores: dict[str, float] = defaultdict(float)
    for rank, fp in enumerate(bm25_top, start=1):
        scores[fp] += alpha * (1.0 / (k + rank))
    for rank, fp in enumerate(dense_top, start=1):
        scores[fp] += (1.0 - alpha) * (1.0 / (k + rank))
    return [fp for fp, _ in sorted(scores.items(), key=lambda x: -x[1])]


def ndcg10(ranked: list[str], gold: set[str]) -> float:
    """Binary-relevance nDCG@10 with log-2 discount."""
    dcg = 0.0
    for i, fp in enumerate(ranked[:10], start=1):
        if fp in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold), 10)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg


def evaluate_held_out_repo(
    held_out: str,
    cal_router: Router,
    smoke_root: Path,
    qrel_dir: Path,
    smoke_subdir: str,
    fixed_alphas: list[float],
    theta_hard: float = float("inf"),
) -> dict:
    smoke_dir = smoke_root / smoke_subdir / held_out
    smoke_files = list(smoke_dir.glob("smoke_test_*.jsonl")) if smoke_dir.is_dir() else []
    if not smoke_files:
        return {"repo": held_out, "n": 0}
    qrel_path = qrel_dir / f"{held_out}.jsonl"
    if not qrel_path.exists():
        return {"repo": held_out, "n": 0}
    queries = load_queries(qrel_path)

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    per_query: list[dict] = []

    for line in smoke_files[0].read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        qid = r["query_id"]
        if qid not in queries:
            continue
        gold = set(r.get("gold_files", []))
        if not gold:
            continue
        bm25_top = r.get("bm25_top10", [])
        dense_top = r.get("dense_top10", [])
        qtext = queries[qid]
        alpha_router = cal_router.alpha_blend(qtext)
        tau_hat = cal_router.tau_query(qtext)
        nd_dense = ndcg10(dense_top, gold)
        nd_bm25 = ndcg10(bm25_top, gold)
        rec = {
            "qid": qid,
            "alpha_router": alpha_router,
            "tau_hat": tau_hat,
            "ndcg_dense": nd_dense,
            "ndcg_bm25": nd_bm25,
            "ndcg_router": ndcg10(rrf_fuse(bm25_top, dense_top, alpha_router), gold),
            "ndcg_hard": nd_bm25 if tau_hat >= theta_hard else nd_dense,
        }
        for a in fixed_alphas:
            rec[f"ndcg_a{a:.2f}"] = ndcg10(rrf_fuse(bm25_top, dense_top, a), gold)
        per_query.append(rec)
        for key, val in rec.items():
            if key.startswith("ndcg_"):
                method = key[len("ndcg_"):]
                sums[method] += val
                counts[method] += 1

    out = {"repo": held_out, "n": len(per_query)}
    for method, s in sums.items():
        c = counts.get(method, 0)
        out[f"mean_ndcg_{method}"] = s / c if c else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["A", "B"], required=True)
    ap.add_argument("--clones", type=Path,
                    default=(DATA_ROOT / 'repos'))
    ap.add_argument("--smoke-root", type=Path,
                    default=(DATA_ROOT / 'smoke_test_results'))
    ap.add_argument("--qrel-dir", type=Path,
                    default=(DATA_ROOT / 'qrels'))
    ap.add_argument("--qrel-dir-cellB", type=Path,
                    default=(DATA_ROOT / 'qrels_post_cutoff'))
    ap.add_argument("--repos", nargs="+", default=None)
    ap.add_argument("--fixed-alphas", nargs="+", type=float, default=[0.0, 0.3, 0.5, 1.0])
    ap.add_argument("--out", type=Path, default=(DATA_ROOT / 'hybrid_eval.json'))
    args = ap.parse_args()

    repos = args.repos or REPOS
    smoke_subdir = "cellA_jina15_c1000" if args.cell == "A" else "cellB_stripped_c1000"
    qdir = args.qrel_dir if args.cell == "A" else args.qrel_dir_cellB

    per_repo: list[dict] = []
    for held_out in repos:
        print(f"[{args.cell}] holding out {held_out}", file=sys.stderr)
        cal_repos = [r for r in repos if r != held_out]
        router = build_router_for_repos(args.clones, cal_repos)
        router, cal_pairs = calibrate_router(
            router, args.smoke_root, qdir, cal_repos, smoke_subdir)
        theta_hard = calibrate_hard_threshold(cal_pairs)
        rec = evaluate_held_out_repo(
            held_out, router, args.smoke_root, qdir, smoke_subdir,
            args.fixed_alphas, theta_hard=theta_hard,
        )
        rec["semantic_threshold"] = router.semantic_threshold
        rec["hybrid_threshold"] = router.hybrid_threshold
        rec["hard_threshold"] = theta_hard if math.isfinite(theta_hard) else None
        per_repo.append(rec)
        print(f"  n={rec.get('n', 0)} "
              f"dense={rec.get('mean_ndcg_dense', 0):.4f} "
              f"bm25={rec.get('mean_ndcg_bm25', 0):.4f} "
              f"router={rec.get('mean_ndcg_router', 0):.4f} "
              f"thr=({rec['semantic_threshold']:.2f},{rec['hybrid_threshold']:.2f})",
              file=sys.stderr)

    # Pool over held-out queries (weighted by query count per repo)
    pooled: dict[str, float] = defaultdict(float)
    total_n = 0
    for r in per_repo:
        n = r.get("n", 0)
        total_n += n
        for key, val in r.items():
            if key.startswith("mean_ndcg_"):
                pooled[key] += val * n
    summary_pooled = {k: (v / total_n) if total_n else 0.0 for k, v in pooled.items()}
    summary_pooled["total_n"] = total_n

    out = {"cell": args.cell, "per_repo": per_repo, "pooled": summary_pooled}
    args.out.write_text(json.dumps(out, indent=2))
    print("=== Pooled (weighted by per-repo n) ===")
    print(json.dumps(summary_pooled, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
