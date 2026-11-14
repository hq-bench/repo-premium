"""Plot the two outstanding figures for the ICML paper:

  Fig 1 — Chunking confound: file-level vs chunked retrieval on django.
          BM25 only mildly affected; dense flips from −0.153 to +0.118.
          (Single repo; file-level was only run on django.)

  Fig 5 — Encoder scaling asymmetric: per-repo Δ for jina-code-0.5b vs 1.5b,
          grouped by cell. Cell A bars are flat; Cell B bars are clearly
          green-shifted (positive uplift) for the bigger model.

Reads per-query JSONL output of smoke_test_django.py:
  - ${REPO_PREMIUM_DATA:-data}/coreb_repo_jsonl/django_filelevel/   (file-level, chunked off)
  - ${REPO_PREMIUM_DATA:-data}/coreb_repo_jsonl/cellA_c1000/        (0.5b at canonical chunk)
  - ${REPO_PREMIUM_DATA:-data}/coreb_repo_jsonl/cellB_c1000/        (0.5b at canonical chunk)
  - ${REPO_PREMIUM_DATA:-data}/coreb_repo_jsonl/cellA_jina15_c1000/ (1.5b at canonical chunk)
  - ${REPO_PREMIUM_DATA:-data}/coreb_repo_jsonl/cellB_jina15_c1000/ (1.5b at canonical chunk)

Writes:
  - figures/coreb_repo/fig1_chunking_confound.png
  - figures/coreb_repo/fig5_encoder_scaling.png
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

CACHE_ROOT = (DATA_ROOT / 'smoke_test_results')
OUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "coreb_repo"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def hide_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_cell(cell: str) -> list[dict]:
    rows: list[dict] = []
    base = CACHE_ROOT / cell
    if not base.exists():
        return rows
    if (base / "smoke_test_django.jsonl").exists():
        with (base / "smoke_test_django.jsonl").open() as fh:
            for line in fh:
                r = json.loads(line)
                r["repo"] = "django__django"
                rows.append(r)
        return rows
    for repo_dir in sorted(base.iterdir()):
        f = repo_dir / "smoke_test_django.jsonl"
        if not f.exists():
            continue
        with f.open() as fh:
            for line in fh:
                r = json.loads(line)
                r["repo"] = repo_dir.name.replace("_c1000", "")
                rows.append(r)
    return rows


def fig1_chunking_confound():
    """Single-repo (django) before/after: file-level retrieval vs chunked.

    Three bars: BM25, dense, Δ — paired (file-level vs chunked).
    """
    file_rows = load_cell("django_filelevel")
    chunk_rows = [r for r in load_cell("cellA_c1000") if r["repo"] == "django__django"]

    if not file_rows or not chunk_rows:
        print(f"[fig1] skipping — file_rows={len(file_rows)}, chunk_rows={len(chunk_rows)}")
        return

    def agg(rows):
        bm25 = np.mean([r["bm25_ndcg10"] for r in rows])
        dense = np.mean([r["dense_ndcg10"] for r in rows])
        delta = np.mean([r["delta_ndcg10"] for r in rows])
        return bm25, dense, delta

    bm25_f, dense_f, delta_f = agg(file_rows)
    bm25_c, dense_c, delta_c = agg(chunk_rows)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    metrics = ["BM25 nDCG@10", "Dense nDCG@10", "Δ = Dense − BM25"]
    vals_file = [bm25_f, dense_f, delta_f]
    vals_chunk = [bm25_c, dense_c, delta_c]

    x = np.arange(len(metrics))
    w = 0.38
    ax.bar(x - w / 2, vals_file, width=w, color="#d62728", alpha=0.85,
           label=f"file-level (n={len(file_rows)})")
    ax.bar(x + w / 2, vals_chunk, width=w, color="#2ca02c", alpha=0.85,
           label=f"chunked c=1000 (n={len(chunk_rows)})")
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)

    # Annotate values
    for i, (vf, vc) in enumerate(zip(vals_file, vals_chunk)):
        ax.text(i - w / 2, vf + (0.012 if vf >= 0 else -0.030),
                f"{vf:+.3f}", ha="center", fontsize=8, color="#d62728")
        ax.text(i + w / 2, vc + (0.012 if vc >= 0 else -0.030),
                f"{vc:+.3f}", ha="center", fontsize=8, color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel("nDCG@10")
    ax.set_title("Chunking is a confound, not a tuning knob (django, n=226)",
                 fontsize=10)
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(-0.30, 0.65)
    hide_spines(ax)

    fig.tight_layout()
    out = OUT_DIR / "fig1_chunking_confound.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  "
          f"(file Δ={delta_f:+.3f} → chunked Δ={delta_c:+.3f}, "
          f"sign {'flip' if delta_f * delta_c < 0 else 'preserved'})")


def fig5_encoder_scaling():
    """Per-repo Δ bars: 0.5b vs 1.5b, Cell A and Cell B side-by-side.

    Two subpanels (Cell A, Cell B). Within each, per-repo paired bars of Δ.
    Cell A bars should be near-zero; Cell B bars should be clearly green
    (1.5b > 0.5b).
    """
    A_05 = load_cell("cellA_c1000")
    A_15 = load_cell("cellA_jina15_c1000")
    B_05 = load_cell("cellB_c1000")
    B_15 = load_cell("cellB_jina15_c1000")

    def by_repo_delta(rows):
        d = defaultdict(list)
        for r in rows:
            d[r["repo"]].append(r["delta_ndcg10"])
        return {k: float(np.mean(v)) for k, v in d.items()}

    def normalize(repo):
        s = repo.replace("_c1000", "")
        s = (s.replace("__", "_")
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
        return s

    A05d = {normalize(k): v for k, v in by_repo_delta(A_05).items()}
    A15d = {normalize(k): v for k, v in by_repo_delta(A_15).items()}
    B05d = {normalize(k): v for k, v in by_repo_delta(B_05).items()}
    B15d = {normalize(k): v for k, v in by_repo_delta(B_15).items()}

    # Micro (n-weighted) pooled means matching the Findings doc numbers.
    A05_mu = float(np.mean([r["delta_ndcg10"] for r in A_05]))
    A15_mu = float(np.mean([r["delta_ndcg10"] for r in A_15]))
    B05_mu = float(np.mean([r["delta_ndcg10"] for r in B_05]))
    B15_mu = float(np.mean([r["delta_ndcg10"] for r in B_15]))

    # Canonical Cell A repo order (alphabetical by short name)
    A_repos = sorted(set(A05d) & set(A15d))
    B_repos = sorted(set(B05d) & set(B15d))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6),
                                   gridspec_kw={"width_ratios": [10, 7]})

    def plot_cell(ax, repos, d05, d15, title, ylim, m05, m15):
        x = np.arange(len(repos))
        w = 0.38
        v05 = [d05[r] for r in repos]
        v15 = [d15[r] for r in repos]
        ax.bar(x - w / 2, v05, width=w, color="#1f77b4", alpha=0.85,
               label="jina-code-0.5b")
        ax.bar(x + w / 2, v15, width=w, color="#9467bd", alpha=0.85,
               label="jina-code-1.5b")
        ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(repos, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Δ = nDCG_dense − nDCG_BM25")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper left", frameon=False, fontsize=7.5)
        ax.set_ylim(*ylim)
        hide_spines(ax)
        ax.text(0.98, 0.95,
                f"pooled Δ:  0.5b={m05:+.3f}\n            1.5b={m15:+.3f}\n            uplift={m15-m05:+.3f}",
                transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
                bbox=dict(facecolor="white", edgecolor="lightgray",
                          boxstyle="round,pad=0.3"))

    plot_cell(ax1, A_repos, A05d, A15d,
              "Cur-Pre (pre-cutoff, n=472): scaling is flat",
              (-0.15, 0.60), A05_mu, A15_mu)
    plot_cell(ax2, B_repos, B05d, B15d,
              "Raw-Post (post-cutoff, n=140): scaling lifts uniformly",
              (-0.15, 0.85), B05_mu, B15_mu)

    fig.tight_layout()
    out = OUT_DIR / "fig5_encoder_scaling.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    A_uplift = np.mean(list(A15d.values())) - np.mean(list(A05d.values()))
    B_uplift = np.mean(list(B15d.values())) - np.mean(list(B05d.values()))
    print(f"wrote {out}  "
          f"(A uplift={A_uplift:+.3f}, B uplift={B_uplift:+.3f}, "
          f"ratio {B_uplift/A_uplift:.1f}×)")


if __name__ == "__main__":
    fig1_chunking_confound()
    fig5_encoder_scaling()
