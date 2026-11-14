"""Round-trip validation of embedding_premium.Router against §5.3 per-query Δ.

For each of the 10 canonical repos:
  1. Build Router from chunks of the clone (c=1000 with 10-line overlap),
     matching the smoke-test corpus exactly.
  2. Load smoke-test per-query records (have Δ and offline τ_q).
  3. Load qrels (have query text).
  4. Join on query_id, run router.predict(query) per query.
  5. Aggregate Δ per predicted regime.

The router's deployment-time predict() should reproduce the §5.3 quartile
ordering: lexical (BM25 alone) regime should have the lowest mean Δ,
semantic regime the highest.

CPU-only. Run inside the Apptainer container or with system python.
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from embedding_premium import Router  # noqa: E402
from run_retrieval import (  # noqa: E402
    chunk_file,
    list_corpus,
    read_file,
    strip_pr_boilerplate,
)

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


def build_router(clone_dir: Path) -> Router:
    paths = list_corpus(clone_dir)
    chunks: list[str] = []
    for rel in paths:
        text = read_file(clone_dir, rel)
        if not text.strip():
            continue
        chunks.extend(chunk_file(text, target_chars=1000, overlap_lines=10))
    return Router.fit(chunks)


def load_queries(qrel_path: Path) -> dict[str, str]:
    """query_id → problem_statement (raw, unstripped)."""
    out: dict[str, str] = {}
    for line in qrel_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["query_id"]] = r.get("problem_statement", "") or r.get("title", "")
    return out


def load_smoke(smoke_path: Path) -> list[dict]:
    return [json.loads(l) for l in smoke_path.read_text().splitlines() if l.strip()]


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
    ap.add_argument("--out", type=Path,
                    default=(DATA_ROOT / 'router_validation.json'))
    ap.add_argument("--repos", nargs="+", default=None,
                    help="Subset of repos to evaluate (default: all 10)")
    args = ap.parse_args()
    repos_to_run = args.repos or REPOS

    smoke_subdir = "cellA_jina15_c1000" if args.cell == "A" else "cellB_stripped_c1000"
    qdir = args.qrel_dir if args.cell == "A" else args.qrel_dir_cellB

    # regime → list[Δ]
    pool: dict[str, list[float]] = defaultdict(list)
    # regime → list[(tau_q, mean_idf)] for diagnostics
    diag: dict[str, list[tuple[float, float]]] = defaultdict(list)
    per_repo: dict[str, dict] = {}
    # all (router_idf, delta, paper_tau, has_ids) for distribution analysis
    all_points: list[tuple[float, float, float, bool]] = []

    for repo in repos_to_run:
        smoke_path_dir = args.smoke_root / smoke_subdir / repo
        if not smoke_path_dir.is_dir():
            continue
        smoke_files = list(smoke_path_dir.glob("smoke_test_*.jsonl"))
        if not smoke_files:
            continue
        smoke_path = smoke_files[0]
        qrel_path = qdir / f"{repo}.jsonl"
        if not qrel_path.exists():
            continue
        clone_dir = args.clones / repo
        if not clone_dir.is_dir():
            continue

        print(f"[{args.cell}] building router for {repo}...", file=sys.stderr)
        router = build_router(clone_dir)
        queries = load_queries(qrel_path)
        recs = load_smoke(smoke_path)

        repo_pool: dict[str, list[float]] = defaultdict(list)
        for r in recs:
            qid = r["query_id"]
            if qid not in queries:
                continue
            qtext = queries[qid]
            if not qtext.strip():
                continue
            regime = router.predict(qtext)
            d = float(r["delta_ndcg10"])
            tau_q = float(r.get("tau_q", 0.0))
            m_idf = router.tau_query(qtext)
            pool[regime].append(d)
            diag[regime].append((tau_q, m_idf))
            repo_pool[regime].append(d)
            all_points.append((m_idf, d, tau_q, router.has_identifiers(qtext)))

        per_repo[repo] = {
            "n_total": sum(len(v) for v in repo_pool.values()),
            "by_regime": {k: {"n": len(v),
                              "mean_delta": float(sum(v) / len(v)) if v else 0.0}
                          for k, v in repo_pool.items()},
            "n_chunks": router.n_chunks,
            "vocab": len(router.idf),
        }

    # Quartile-bin Δ on the router's deployment-time τ (mean_router_idf).
    # If F1 is robust under the deployment proxy, Δ should fall monotonically
    # from Q1 (lowest router_idf) to Q4 (highest).
    quartile_bins: dict[str, dict] = {}
    if all_points:
        import numpy as np
        arr = np.array([(m, d) for m, d, _, _ in all_points])
        m_vals, d_vals = arr[:, 0], arr[:, 1]
        cuts = np.quantile(m_vals, [0.25, 0.5, 0.75])
        for label, lo, hi in [
            ("Q1", -1e9, cuts[0]),
            ("Q2", cuts[0], cuts[1]),
            ("Q3", cuts[1], cuts[2]),
            ("Q4", cuts[2], 1e9),
        ]:
            sel = (m_vals > lo) & (m_vals <= hi)
            if not sel.any():
                continue
            quartile_bins[label] = {
                "n": int(sel.sum()),
                "router_idf_range": [float(m_vals[sel].min()),
                                      float(m_vals[sel].max())],
                "mean_delta": float(d_vals[sel].mean()),
            }
        # Pearson r between m_vals and d_vals
        try:
            from scipy.stats import pearsonr, spearmanr  # type: ignore
            quartile_bins["pearson_r_p"] = list(map(float, pearsonr(m_vals, d_vals)))
            quartile_bins["spearman_rho_p"] = list(map(float, spearmanr(m_vals, d_vals)))
        except Exception:
            r = float(np.corrcoef(m_vals, d_vals)[0, 1])
            quartile_bins["pearson_r"] = r

    summary = {
        "cell": args.cell,
        "quartile_bins": quartile_bins,
        "regime_pool": {k: {
            "n": len(v),
            "mean_delta": float(sum(v) / len(v)) if v else 0.0,
            "mean_tau_q": float(sum(t for t, _ in diag[k]) / len(diag[k])) if diag[k] else 0.0,
            "mean_router_idf": float(sum(m for _, m in diag[k]) / len(diag[k])) if diag[k] else 0.0,
        } for k, v in pool.items()},
        "per_repo": per_repo,
    }

    args.out.write_text(json.dumps(summary, indent=2))
    print("=== Router-IDF quartile bins (deployment-time τ proxy) ===")
    print(json.dumps(quartile_bins, indent=2))
    print("=== Regime pool ===")
    print(json.dumps(summary["regime_pool"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
