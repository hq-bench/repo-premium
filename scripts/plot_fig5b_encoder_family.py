"""Fig 5b — two-family encoder scaling replication (jina-code vs Qwen3).

Plots Cell A and Cell B per-repo Δ uplift (= Δ_big − Δ_small) for two encoder
families side-by-side:

  - jina-code: 0.5b → 1.5b   (existing data from cellA_jina15_c1000 / cellB_jina15_c1000)
  - Qwen3:     0.6B → 4B     (R4 data from cellA_qwen{06,4}_c1000 / cellB_qwen{06,4}_c1000)

Decision rule (pre-registered): F3 survives as "encoder scaling (general)" if Cell B
uplift > Cell A uplift for both families with same sign. Otherwise F3 has to be
qualified.

Reads per-query JSONL output of smoke_test_django.py from ${REPO_PREMIUM_DATA:-data}/coreb_repo_jsonl/.
Writes figures/coreb_repo/fig5b_encoder_family.png.
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


def normalize_repo(repo: str) -> str:
    s = repo.replace("_c1000", "")
    return (s.replace("__", "_")
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


def by_repo_delta(rows: list[dict]) -> dict[str, float]:
    d: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        d[normalize_repo(r["repo"])].append(r["delta_ndcg10"])
    return {k: float(np.mean(v)) for k, v in d.items()}


def micro_delta(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([r["delta_ndcg10"] for r in rows]))


def plot_panel(ax, repos, jina_uplift, qwen_uplift, title, ylim,
               jina_micro, qwen_micro):
    x = np.arange(len(repos))
    w = 0.38
    ax.bar(x - w / 2, [jina_uplift.get(r, 0.0) for r in repos],
           width=w, color="#9467bd", alpha=0.85,
           label=r"jina-code  $\Delta_{1.5b} - \Delta_{0.5b}$")
    ax.bar(x + w / 2, [qwen_uplift.get(r, 0.0) for r in repos],
           width=w, color="#e377c2", alpha=0.85,
           label=r"Qwen3-Emb  $\Delta_{4B} - \Delta_{0.6B}$")
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(repos, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"per-repo uplift  $\Delta_{big} - \Delta_{small}$")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    ax.set_ylim(*ylim)
    hide_spines(ax)
    ax.text(0.98, 0.95,
            f"pooled uplift:\n  jina-code  ={jina_micro:+.3f}\n  Qwen3-Emb  ={qwen_micro:+.3f}",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="lightgray",
                      boxstyle="round,pad=0.3"))


def main():
    # jina-code (already on disk)
    A_jina_s = load_cell("cellA_c1000")
    A_jina_b = load_cell("cellA_jina15_c1000")
    B_jina_s = load_cell("cellB_c1000")
    B_jina_b = load_cell("cellB_jina15_c1000")

    # Qwen3-Embedding (R4 new)
    A_qwen_s = load_cell("cellA_qwen06_c1000")
    A_qwen_b = load_cell("cellA_qwen4_c1000")
    B_qwen_s = load_cell("cellB_qwen06_c1000")
    B_qwen_b = load_cell("cellB_qwen4_c1000")

    if not (A_qwen_s and A_qwen_b and B_qwen_s and B_qwen_b):
        print("[fig5b] Qwen3 cells incomplete — skipping. "
              f"A_qwen_s={len(A_qwen_s)} A_qwen_b={len(A_qwen_b)} "
              f"B_qwen_s={len(B_qwen_s)} B_qwen_b={len(B_qwen_b)}")
        return

    # Per-repo Δ then per-repo uplift = big − small
    A_jina_uplift = {
        r: by_repo_delta(A_jina_b).get(r, np.nan)
           - by_repo_delta(A_jina_s).get(r, np.nan)
        for r in sorted(set(by_repo_delta(A_jina_s)) & set(by_repo_delta(A_jina_b)))
    }
    A_qwen_uplift = {
        r: by_repo_delta(A_qwen_b).get(r, np.nan)
           - by_repo_delta(A_qwen_s).get(r, np.nan)
        for r in sorted(set(by_repo_delta(A_qwen_s)) & set(by_repo_delta(A_qwen_b)))
    }
    B_jina_uplift = {
        r: by_repo_delta(B_jina_b).get(r, np.nan)
           - by_repo_delta(B_jina_s).get(r, np.nan)
        for r in sorted(set(by_repo_delta(B_jina_s)) & set(by_repo_delta(B_jina_b)))
    }
    B_qwen_uplift = {
        r: by_repo_delta(B_qwen_b).get(r, np.nan)
           - by_repo_delta(B_qwen_s).get(r, np.nan)
        for r in sorted(set(by_repo_delta(B_qwen_s)) & set(by_repo_delta(B_qwen_b)))
    }

    A_repos = sorted(set(A_jina_uplift) & set(A_qwen_uplift))
    B_repos = sorted(set(B_jina_uplift) & set(B_qwen_uplift))

    # Pooled micro-uplift (per-query average of big − per-query average of small)
    A_jina_micro = micro_delta(A_jina_b) - micro_delta(A_jina_s)
    A_qwen_micro = micro_delta(A_qwen_b) - micro_delta(A_qwen_s)
    B_jina_micro = micro_delta(B_jina_b) - micro_delta(B_jina_s)
    B_qwen_micro = micro_delta(B_qwen_b) - micro_delta(B_qwen_s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8),
                                   gridspec_kw={"width_ratios": [10, 7]})
    plot_panel(ax1, A_repos, A_jina_uplift, A_qwen_uplift,
               "Cell A (pre-cutoff, n=472): scaling flat for both families",
               (-0.10, 0.20), A_jina_micro, A_qwen_micro)
    plot_panel(ax2, B_repos, B_jina_uplift, B_qwen_uplift,
               "Cell B (post-cutoff, n=139): post-cutoff lift in both families",
               (-0.10, 0.20), B_jina_micro, B_qwen_micro)

    fig.tight_layout()
    out = OUT_DIR / "fig5b_encoder_family.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  Cell A jina uplift = {A_jina_micro:+.4f}, qwen uplift = {A_qwen_micro:+.4f}")
    print(f"  Cell B jina uplift = {B_jina_micro:+.4f}, qwen uplift = {B_qwen_micro:+.4f}")
    print(f"  jina B/A ratio = {B_jina_micro/A_jina_micro:+.2f}×  "
          f"qwen B/A ratio = {B_qwen_micro/A_qwen_micro:+.2f}×")


if __name__ == "__main__":
    main()
