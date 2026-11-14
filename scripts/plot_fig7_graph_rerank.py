"""Fig 7 — graph-as-rerank on dense top-K, by cell and encoder.

Tests whether layering a graph-centrality signal on top of dense retrieval
amplifies or substitutes for the embedding premium.

Reads:
  ${REPO_PREMIUM_DATA:-data}/coreb_repo_graph_rerank/{cellA,cellB}/{encoder}/{repo}.jsonl
  (graph_rerank.py output: dense_ndcg10 + graph_rerank_ndcg10 per query)

Writes:
  figures/coreb_repo/fig7_graph_rerank.png

Two-panel layout:
  Top:    paired bars per (cell × encoder) — dense vs dense+rerank, nDCG@10.
  Bottom: per-query win/loss bars (better − worse) per (cell × encoder).
"""
from __future__ import annotations
from paths import DATA_ROOT

import json
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
ENCODERS = [("jina-0.5b", "jina-code-0.5b"),
            ("jina-1.5b", "jina-code-1.5b"),
            ("Qwen3-0.6B", "Qwen3-Emb-0.6B"),
            ("Qwen3-4B", "Qwen3-Emb-4B")]

ROOT = (DATA_ROOT / 'coreb_repo_graph_rerank')
OUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "coreb_repo"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rows(cell: str, model_dir: str, repo: str) -> list[dict]:
    p = ROOT / cell / model_dir / f"{repo}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def pooled(cell: str, repos: list[str], model_dir: str) -> dict | None:
    dn, rn = [], []
    better = worse = same = 0
    for r in repos:
        for x in load_rows(cell, model_dir, r):
            dn.append(x["dense_ndcg10"])
            rn.append(x["graph_rerank_ndcg10"])
            if x["graph_rerank_ndcg10"] > x["dense_ndcg10"]:
                better += 1
            elif x["graph_rerank_ndcg10"] < x["dense_ndcg10"]:
                worse += 1
            else:
                same += 1
    if not dn:
        return None
    n = len(dn)
    return {"n": n, "dense": sum(dn) / n, "rerank": sum(rn) / n,
            "delta": (sum(rn) - sum(dn)) / n,
            "better": better, "worse": worse, "same": same}


def hide_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_paired(ax, A_stats, B_stats):
    """Paired bars: dense vs dense+rerank, per (cell × encoder)."""
    cells = ["Cell A", "Cell B"]
    labels = [lab for _, lab in ENCODERS]
    n_cells = 2
    n_enc = len(ENCODERS)
    x = np.arange(n_enc * n_cells)
    w = 0.4

    dense_vals = []
    rerank_vals = []
    deltas = []
    for cell_stats in (A_stats, B_stats):
        for mdir, _ in ENCODERS:
            s = cell_stats[mdir]
            dense_vals.append(s["dense"] if s else 0)
            rerank_vals.append(s["rerank"] if s else 0)
            deltas.append(s["delta"] if s else 0)

    # Color: cool blue for dense, warm orange for rerank.
    ax.bar(x - w / 2, dense_vals, width=w, color="#4c72b0", alpha=0.85,
           label="dense top-10")
    ax.bar(x + w / 2, rerank_vals, width=w, color="#dd8452", alpha=0.9,
           label="dense + graph rerank (α=1.5)")

    # Annotate deltas above the rerank bar.
    for xi, (rv, d) in enumerate(zip(rerank_vals, deltas)):
        sign = "+" if d >= 0 else ""
        color = "#2c8a2c" if d > 0.005 else ("#a62a2a" if d < -0.005 else "#555")
        ax.text(x[xi] + w / 2, rv + 0.012, f"{sign}{d:.3f}",
                ha="center", va="bottom", fontsize=7.5, color=color,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels * n_cells, rotation=25, ha="right", fontsize=8)
    # Cell separator + label.
    ax.axvline(n_enc - 0.5, color="#999", lw=0.6, ls="--")
    n_A = next(iter(A_stats.values()))["n"] if A_stats else 0
    n_B = next(iter(B_stats.values()))["n"] if B_stats else 0
    ax.text(np.mean(x[:n_enc]), ax.get_ylim()[1] * 0.96,
            f"Cell A (curated, n={n_A})", ha="center", fontsize=9,
            fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(np.mean(x[n_enc:]), 0.96,
            f"Cell B (real PRs, n={n_B})", ha="center", fontsize=9,
            fontweight="bold", transform=ax.transAxes)
    ax.set_ylabel("nDCG@10 (pooled across repos)")
    ax.set_ylim(0, max(max(dense_vals), max(rerank_vals)) * 1.18)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Graph-as-rerank on dense top-10 — uniform lift on Cell A, "
                 "neutral on Cell B", fontsize=10)
    hide_spines(ax)


def plot_winloss(ax, A_stats, B_stats):
    """Per-query better vs worse counts, normalized."""
    cells = [("Cell A", A_stats), ("Cell B", B_stats)]
    labels = [lab for _, lab in ENCODERS]
    x = np.arange(len(ENCODERS) * len(cells))
    w = 0.4

    better_frac = []
    worse_frac = []
    for _, stats in cells:
        for mdir, _ in ENCODERS:
            s = stats[mdir]
            if not s or s["n"] == 0:
                better_frac.append(0)
                worse_frac.append(0)
            else:
                better_frac.append(s["better"] / s["n"])
                worse_frac.append(-s["worse"] / s["n"])

    ax.bar(x, better_frac, width=w * 2, color="#2ca02c", alpha=0.85,
           label="rerank better")
    ax.bar(x, worse_frac, width=w * 2, color="#d62728", alpha=0.75,
           label="rerank worse")
    for xi, (b, w_) in enumerate(zip(better_frac, worse_frac)):
        ax.text(x[xi], b + 0.015, f"{int(b * 100)}%",
                ha="center", va="bottom", fontsize=7.5)
        ax.text(x[xi], w_ - 0.015, f"{int(-w_ * 100)}%",
                ha="center", va="top", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels * len(cells), rotation=25, ha="right", fontsize=8)
    ax.axvline(len(ENCODERS) - 0.5, color="#999", lw=0.6, ls="--")
    ax.axhline(0, color="#333", lw=0.7)
    ax.set_ylabel("share of queries (rerank vs dense)")
    ax.set_ylim(min(worse_frac) * 1.4 - 0.05, max(better_frac) * 1.3 + 0.05)
    ax.set_title("Per-query win/loss — 3-5:1 on Cell A, 1:1 on Cell B",
                 fontsize=10)
    ax.legend(loc="upper right", frameon=False)
    # Cell labels.
    ax.text(np.mean(x[:len(ENCODERS)]), 0.96, "Cell A",
            ha="center", fontsize=9, fontweight="bold",
            transform=ax.transAxes)
    ax.text(np.mean(x[len(ENCODERS):]), 0.96, "Cell B",
            ha="center", fontsize=9, fontweight="bold",
            transform=ax.transAxes)
    hide_spines(ax)


def main():
    A = {mdir: pooled("cellA", REPOS_A, mdir) for mdir, _ in ENCODERS}
    B = {mdir: pooled("cellB", REPOS_B, mdir) for mdir, _ in ENCODERS}

    fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                              gridspec_kw={"height_ratios": [1.2, 1]})
    plot_paired(axes[0], A, B)
    plot_winloss(axes[1], A, B)
    fig.suptitle("Fig 7 — Graph rerank is a task-shape proxy: amplifies on "
                  "curated, neutral on real PRs",
                  fontsize=11, y=1.00)
    fig.tight_layout()
    out = OUT_DIR / "fig7_graph_rerank.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    print("\n=== Cell A pooled (per encoder) ===")
    for mdir, lab in ENCODERS:
        s = A[mdir]
        if s:
            print(f"  {lab:<18s} n={s['n']:>4d}  dense={s['dense']:.4f}  "
                  f"rerank={s['rerank']:.4f}  Δ={s['delta']:+.4f}  "
                  f"({s['better']} better / {s['worse']} worse)")
    print("\n=== Cell B pooled (per encoder) ===")
    for mdir, lab in ENCODERS:
        s = B[mdir]
        if s:
            print(f"  {lab:<18s} n={s['n']:>4d}  dense={s['dense']:.4f}  "
                  f"rerank={s['rerank']:.4f}  Δ={s['delta']:+.4f}  "
                  f"({s['better']} better / {s['worse']} worse)")


if __name__ == "__main__":
    main()
