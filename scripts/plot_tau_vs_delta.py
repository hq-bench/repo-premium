"""Plot τ(q) vs Δ scatter + quartile bins + raw-vs-stripped Cell B comparison.

Reads per-query JSONL output of smoke_test_django.py (pulled to ${REPO_PREMIUM_DATA:-data}/coreb_repo_jsonl/).
Writes 3 figures to figures/coreb_repo/.

Usage:
  conda run -n llm python scripts/plot_tau_vs_delta.py
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
from scipy import stats

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

CACHE_ROOT = (DATA_ROOT / 'smoke_test_results')
OUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "coreb_repo"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cell(cell: str) -> list[dict]:
    rows: list[dict] = []
    base = CACHE_ROOT / cell
    for repo_dir in sorted(base.iterdir()):
        f = repo_dir / "smoke_test_django.jsonl"
        if not f.exists():
            continue
        with f.open() as fh:
            for line in fh:
                r = json.loads(line)
                r["repo"] = repo_dir.name
                rows.append(r)
    return rows


def hide_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def scatter_plus_quartiles(rows: list[dict], title: str, out_path: Path, color: str = "#1f77b4"):
    tau = np.array([r["tau_q"] for r in rows])
    delta = np.array([r["delta_ndcg10"] for r in rows])

    # Quartile bin means
    q_edges = np.quantile(tau, [0, 0.25, 0.5, 0.75, 1.0])
    bin_idx = np.clip(np.digitize(tau, q_edges[1:-1]), 0, 3)
    bin_means = [delta[bin_idx == i].mean() for i in range(4)]
    bin_centers = [(q_edges[i] + q_edges[i + 1]) / 2 for i in range(4)]

    pearson_r, pearson_p = stats.pearsonr(tau, delta)
    spearman_r, _ = stats.spearmanr(tau, delta)

    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    ax.scatter(tau, delta, s=12, alpha=0.25, color=color, edgecolors="none", label=f"queries (n={len(rows)})")
    ax.plot(bin_centers, bin_means, "o-", color="#d62728", markersize=8, lw=2.0,
            label="quartile mean Δ")
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)

    ax.set_xlabel(r"$\tau(q)$ — lexical distinctiveness (mean IDF, top-5)")
    ax.set_ylabel(r"$\Delta$ = nDCG$_{dense}$ − nDCG$_{BM25}$")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", frameon=False)
    txt = f"Pearson r = {pearson_r:+.3f}  (p = {pearson_p:.3f})\nSpearman ρ = {spearman_r:+.3f}\n"
    txt += "Q1→Q4 Δ: " + " → ".join(f"{m:+.3f}" for m in bin_means)
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=7.5,
            verticalalignment="bottom",
            bbox=dict(facecolor="white", edgecolor="lightgray", boxstyle="round,pad=0.3"))
    hide_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  (n={len(rows)}, r={pearson_r:+.3f}, p={pearson_p:.3f})")


def scatter_cellAB(rowsA: list[dict], rowsB: list[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for rows, color, label in [
        (rowsA, "#1f77b4", f"Cell A (curated SWE-Bench, n={len(rowsA)})"),
        (rowsB, "#ff7f0e", f"Cell B (mined PRs, n={len(rowsB)})"),
    ]:
        tau = np.array([r["tau_q"] for r in rows])
        delta = np.array([r["delta_ndcg10"] for r in rows])
        ax.scatter(tau, delta, s=14, alpha=0.30, color=color, edgecolors="none", label=label)
        # Quartile means (per-cell)
        q_edges = np.quantile(tau, [0, 0.25, 0.5, 0.75, 1.0])
        bin_idx = np.clip(np.digitize(tau, q_edges[1:-1]), 0, 3)
        bin_centers = [(q_edges[i] + q_edges[i + 1]) / 2 for i in range(4)]
        bin_means = [delta[bin_idx == i].mean() if (bin_idx == i).any() else np.nan
                     for i in range(4)]
        ax.plot(bin_centers, bin_means, "o-", color=color, markersize=8, lw=2.2,
                markeredgecolor="black", markeredgewidth=0.5)
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel(r"$\tau(q)$ — lexical distinctiveness (mean IDF, top-5)")
    ax.set_ylabel(r"$\Delta$ = nDCG$_{dense}$ − nDCG$_{BM25}$")
    ax.set_title("Embedding premium: curated vs mined queries (boilerplate stripped, c=1000)", fontsize=10)
    ax.legend(loc="upper right", frameon=False)
    hide_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def cellB_raw_vs_stripped(rowsB_raw: list[dict], rowsB_stp: list[dict], out_path: Path):
    """Per-repo BM25 nDCG raw vs stripped + Δ raw vs stripped."""
    repos = sorted({r["repo"] for r in rowsB_raw} & {r["repo"] for r in rowsB_stp})

    def by_repo(rows):
        out = defaultdict(list)
        for r in rows:
            out[r["repo"]].append(r)
        return out

    raw = by_repo(rowsB_raw)
    stp = by_repo(rowsB_stp)

    bm25_raw = [np.mean([r["bm25_ndcg10"] for r in raw[k]]) for k in repos]
    bm25_stp = [np.mean([r["bm25_ndcg10"] for r in stp[k]]) for k in repos]
    delta_raw = [np.mean([r["delta_ndcg10"] for r in raw[k]]) for k in repos]
    delta_stp = [np.mean([r["delta_ndcg10"] for r in stp[k]]) for k in repos]

    short = [r.split("__")[-1].replace("scikit-learn", "sklearn").replace("matplotlib", "mpl")[:8] for r in repos]
    x = np.arange(len(repos))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.4))

    ax1.bar(x - w / 2, bm25_raw, width=w, color="#d62728", label="raw", alpha=0.85)
    ax1.bar(x + w / 2, bm25_stp, width=w, color="#2ca02c", label="stripped", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, rotation=0, fontsize=8)
    ax1.set_ylabel("BM25 nDCG@10")
    ax1.set_title("Cell B: BM25 recovers when PR-boilerplate is stripped", fontsize=10)
    ax1.legend(loc="upper left", frameon=False)
    ax1.set_ylim(0, 0.75)
    hide_spines(ax1)

    ax2.bar(x - w / 2, delta_raw, width=w, color="#d62728", label="raw", alpha=0.85)
    ax2.bar(x + w / 2, delta_stp, width=w, color="#2ca02c", label="stripped", alpha=0.85)
    ax2.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, rotation=0, fontsize=8)
    ax2.set_ylabel(r"$\Delta$ = nDCG$_{dense}$ − nDCG$_{BM25}$")
    ax2.set_title("Cell B: embedding premium shrinks after stripping", fontsize=10)
    ax2.legend(loc="upper right", frameon=False)
    hide_spines(ax2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    # canonical = chunk=1000
    rowsA = load_cell("cellA_c1000")
    rowsB_stp = load_cell("cellB_c1000")
    rowsB_raw = load_cell("cellB_chunked")

    scatter_plus_quartiles(
        rowsA,
        title="Cell A — τ(q) vs embedding premium (canonical: c=1000, stripped)",
        out_path=OUT_DIR / "tau_vs_delta_cellA_canonical.png",
        color="#1f77b4",
    )
    scatter_cellAB(rowsA, rowsB_stp, OUT_DIR / "tau_vs_delta_cellAB_canonical.png")
    cellB_raw_vs_stripped(rowsB_raw, rowsB_stp, OUT_DIR / "cellB_raw_vs_stripped_canonical.png")


if __name__ == "__main__":
    main()
