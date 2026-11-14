"""Fig 6 — code-graph baseline vs BM25 vs dense, by cell.

Tests the reviewer claim "code graph is the SOTA for code search." Two panels:

  Cell A (curated SWE-Bench Verified, n=472): identifier-rich issues — graph
      search is competitive with dense.
  Cell B (post-cutoff PRs, n=139): realistic queries — graph collapses,
      dense opens a 27-pt lead.

Reads:
  - ${REPO_PREMIUM_DATA:-data}/coreb_repo_graph_eval/{cellA,cellB}/<repo>.jsonl (graph_search.py output)
  - ${REPO_PREMIUM_DATA:-data}/coreb_repo_jsonl/{cell}_{model}/<repo>/smoke_test_django.jsonl
                                                   (BM25 + dense from R4 jsonls)

Writes:
  figures/coreb_repo/fig6_graph_baseline.png
"""
from __future__ import annotations
from paths import DATA_ROOT
from paths import CACHE_ROOT

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

REPOS_A = ["astropy__astropy", "django__django", "matplotlib__matplotlib",
           "psf__requests", "pydata__xarray", "pylint-dev__pylint",
           "pytest-dev__pytest", "scikit-learn__scikit-learn",
           "sphinx-doc__sphinx", "sympy__sympy"]
REPOS_B = ["astropy__astropy", "django__django", "matplotlib__matplotlib",
           "pydata__xarray", "pytest-dev__pytest",
           "scikit-learn__scikit-learn", "sympy__sympy"]

CACHE_ROOT = (DATA_ROOT / 'smoke_test_results')
GRAPH_ROOT = (DATA_ROOT / 'coreb_repo_graph_eval')
OUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "coreb_repo"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _dense_sub(cell: str, model_dir: str, repo: str) -> str:
    if cell == "cellA" and model_dir == "c1000":
        return f"{repo}_c1000"
    return repo


def load_dense(cell: str, model_dir: str, repo: str) -> dict[str, dict]:
    sub = _dense_sub(cell, model_dir, repo)
    p = CACHE_ROOT / f"{cell}_{model_dir}" / sub / "smoke_test_django.jsonl"
    if not p.exists():
        return {}
    return {json.loads(l)["query_id"]: json.loads(l) for l in open(p)}


def load_graph(cell: str, repo: str) -> dict[str, dict]:
    p = GRAPH_ROOT / cell / f"{repo}.jsonl"
    if not p.exists():
        return {}
    return {json.loads(l)["query_id"]: json.loads(l) for l in open(p)}


def repo_short(r: str) -> str:
    return (r.replace("__", "_")
             .replace("scikit-learn_scikit-learn", "sklearn")
             .replace("matplotlib_matplotlib", "mpl")
             .replace("pydata_xarray", "xarray")
             .replace("astropy_astropy", "astropy")
             .replace("pytest-dev_pytest", "pytest")
             .replace("sphinx-doc_sphinx", "sphinx")
             .replace("pylint-dev_pylint", "pylint")
             .replace("psf_requests", "requests")
             .replace("django_django", "django")
             .replace("sympy_sympy", "sympy"))


def pooled(cell: str, repos: list[str]) -> dict:
    per_repo = {}
    pooled_lists: dict[str, list[float]] = defaultdict(list)
    MODELS = [("c1000", "jina-0.5b"), ("jina15_c1000", "jina-1.5b"),
              ("qwen06_c1000", "Qwen3-0.6B"), ("qwen4_c1000", "Qwen3-4B")]
    for repo in repos:
        graph = load_graph(cell, repo)
        base = load_dense(cell, "c1000", repo)
        common = set(graph) & set(base)
        if not common:
            continue
        row = {"n": len(common)}
        row["graph"] = float(np.mean([graph[q]["graph_ndcg10"] for q in common]))
        row["bm25"] = float(np.mean([base[q]["bm25_ndcg10"] for q in common]))
        row["grep"] = float(np.mean([base[q]["grep_ndcg10"] for q in common]))
        for mdir, label in MODELS:
            drows = load_dense(cell, mdir, repo)
            vals = [drows[q]["dense_ndcg10"] for q in common if q in drows]
            if vals:
                row[f"dense_{label}"] = float(np.mean(vals))
        per_repo[repo_short(repo)] = row
        for q in common:
            pooled_lists["graph"].append(graph[q]["graph_ndcg10"])
            pooled_lists["bm25"].append(base[q]["bm25_ndcg10"])
            pooled_lists["grep"].append(base[q]["grep_ndcg10"])
            for mdir, label in MODELS:
                drows = load_dense(cell, mdir, repo)
                if q in drows:
                    pooled_lists[f"dense_{label}"].append(drows[q]["dense_ndcg10"])
    pooled_means = {k: float(np.mean(v)) for k, v in pooled_lists.items()}
    pooled_means["n"] = len(pooled_lists["graph"])
    return {"per_repo": per_repo, "pooled": pooled_means}


def hide_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_panel_bars(ax, summary, title):
    """Pooled bar chart: grep, BM25, graph, dense (jina + Qwen3) side-by-side."""
    p = summary["pooled"]
    bars = [
        ("grep", p["grep"], "#bdbdbd"),
        ("BM25", p["bm25"], "#9e9e9e"),
        ("graph (pure)", p["graph"], "#2ca02c"),
        ("Qwen3-0.6B", p.get("dense_Qwen3-0.6B", 0), "#ff7f0e"),
        ("Qwen3-4B", p.get("dense_Qwen3-4B", 0), "#d62728"),
        ("jina-code-0.5b", p.get("dense_jina-0.5b", 0), "#1f77b4"),
        ("jina-code-1.5b", p.get("dense_jina-1.5b", 0), "#17becf"),
    ]
    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    x = np.arange(len(bars))
    rects = ax.bar(x, vals, color=colors, alpha=0.88)
    for rect, v in zip(rects, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("macro nDCG@10")
    ax.set_title(f"{title}  (n={p['n']})", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.18)
    hide_spines(ax)


def plot_per_repo(ax, summary, title):
    """Per-repo nDCG@10: graph vs BM25 vs dense-best (jina-1.5b)."""
    repos = list(summary["per_repo"].keys())
    x = np.arange(len(repos))
    w = 0.28
    bm25 = [summary["per_repo"][r]["bm25"] for r in repos]
    graph = [summary["per_repo"][r]["graph"] for r in repos]
    dense = [summary["per_repo"][r].get("dense_jina-1.5b", 0) for r in repos]
    ax.bar(x - w, bm25, width=w, color="#9e9e9e", alpha=0.85, label="BM25")
    ax.bar(x, graph, width=w, color="#2ca02c", alpha=0.85, label="graph (pure)")
    ax.bar(x + w, dense, width=w, color="#17becf", alpha=0.85,
           label="dense (jina-code-1.5b)")
    ax.set_xticks(x)
    ax.set_xticklabels(repos, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("nDCG@10")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=7.5)
    ax.set_ylim(0, 1.0)
    hide_spines(ax)


def main():
    A = pooled("cellA", REPOS_A)
    B = pooled("cellB", REPOS_B)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2),
                              gridspec_kw={"height_ratios": [1, 1]})
    plot_panel_bars(axes[0, 0], A,
                    "Cell A (curated SWE-Bench Verified)")
    plot_panel_bars(axes[0, 1], B,
                    "Cell B (real post-cutoff PRs)")
    plot_per_repo(axes[1, 0], A, "Cell A — per-repo breakdown")
    plot_per_repo(axes[1, 1], B, "Cell B — per-repo breakdown")
    fig.suptitle("Fig 6 — pure code-graph search: competitive on curated, "
                  "collapses on real PRs",
                  fontsize=11, y=1.00)
    fig.tight_layout()
    out = OUT_DIR / "fig6_graph_baseline.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print("\n=== Cell A pooled ===")
    for k, v in A["pooled"].items():
        print(f"  {k:<22s} = {v:.4f}" if isinstance(v, float) else f"  {k:<22s} = {v}")
    print("\n=== Cell B pooled ===")
    for k, v in B["pooled"].items():
        print(f"  {k:<22s} = {v:.4f}" if isinstance(v, float) else f"  {k:<22s} = {v}")


if __name__ == "__main__":
    main()
