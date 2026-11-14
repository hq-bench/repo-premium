"""Hero figure: τ–Δ phase diagram over pooled Cells A+B, 611 queries.

x = MaxIDF (Hauff 2006) of the top-1 rare shared-token; y = Δ = nDCG_dense − nDCG_BM25.
Encodes:
  - all pooled queries (τ>0) as translucent scatter, colored by cell
  - τ=0 degenerate cluster as a separate left strip (n=38)
  - quartile-mean overlay bars with 95% bootstrap CI
  - annotations: Pearson r, permutation p, Q1→Q4 gap

Reads ${REPO_PREMIUM_DATA:-data}/smoke_test_results/canon202607/jina05_cell{A,B}_c1000/*/smoke_test_django.jsonl
      and ${REPO_PREMIUM_DATA:-data}/smoke_test_results/tau_variants_c1000/cell{A,B}/*/tau_variants.jsonl.
Writes figures/coreb_repo/fig_hero_phase_diagram.{pdf,png}.
"""
from __future__ import annotations
from paths import DATA_ROOT
import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[1]
SMOKE = DATA_ROOT / "smoke_test_results"
DENSE = {"cellA": SMOKE / "canon202607" / "jina05_cellA_c1000", "cellB": SMOKE / "canon202607" / "jina05_cellB_c1000"}
TAUV = SMOKE / "tau_variants_c1000"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "coreb_repo"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260704)


def load_rows() -> list[dict]:
    rows = []
    for cell in ("cellA", "cellB"):
        for repo_dir in sorted(p for p in DENSE[cell].iterdir() if p.is_dir()):
            repo = repo_dir.name
            dm = {}
            with (repo_dir / "smoke_test_django.jsonl").open() as fh:
                for line in fh:
                    r = json.loads(line)
                    dm[r["query_id"]] = r
            tv_file = TAUV / cell / repo / "tau_variants.jsonl"
            if not tv_file.exists():
                continue
            with tv_file.open() as fh:
                for line in fh:
                    r = json.loads(line)
                    qid = r["query_id"]
                    if qid not in dm:
                        continue
                    d = dm[qid]
                    rows.append({
                        "cell": cell,
                        "repo": repo,
                        "query_id": qid,
                        "delta": d["delta_ndcg10"],
                        "tau_maxidf": r["tau_top1"],  # canonical henceforth (Hauff MaxIDF)
                        "tau_jaccard": r["tau_jaccard"],
                        "n_shared": r["n_shared"],
                    })
    return rows


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    means = np.array([RNG.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} queries")

    is_pos = np.array([r["tau_maxidf"] > 0 for r in rows])
    pos_rows = [r for r, k in zip(rows, is_pos) if k]
    zero_rows = [r for r, k in zip(rows, is_pos) if not k]
    print(f"τ>0: {len(pos_rows)}, τ=0: {len(zero_rows)}")

    x = np.array([r["tau_maxidf"] for r in pos_rows])
    y = np.array([r["delta"] for r in pos_rows])
    cells = np.array([r["cell"] for r in pos_rows])

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, _ = stats.spearmanr(x, y)

    # Permutation p (10k shuffles) on |r|
    n_perm = 10_000
    perm_rs = np.empty(n_perm)
    for i in range(n_perm):
        perm_rs[i] = stats.pearsonr(x, RNG.permutation(y))[0]
    perm_p = float((np.abs(perm_rs) >= abs(pearson_r)).mean())

    # Quartile bin means with bootstrap CIs
    q_edges = np.quantile(x, [0, 0.25, 0.5, 0.75, 1.0])
    bin_idx = np.clip(np.digitize(x, q_edges[1:-1]), 0, 3)
    bin_means = []
    bin_los = []
    bin_his = []
    bin_ns = []
    bin_centers = []
    for i in range(4):
        m = (bin_idx == i)
        if not m.any():
            continue
        bin_means.append(y[m].mean())
        lo, hi = bootstrap_ci(y[m])
        bin_los.append(lo)
        bin_his.append(hi)
        bin_ns.append(int(m.sum()))
        bin_centers.append((q_edges[i] + q_edges[i + 1]) / 2)

    zero_delta = np.array([r["delta"] for r in zero_rows]) if zero_rows else np.array([])
    zero_mean = zero_delta.mean() if len(zero_delta) else float("nan")
    zero_lo, zero_hi = bootstrap_ci(zero_delta) if len(zero_delta) > 1 else (float("nan"), float("nan"))

    print(f"Pearson r={pearson_r:+.3f} perm p={perm_p:.3f} Spearman={spearman_r:+.3f}; "
          f"quartiles {[round(m, 3) for m in bin_means]}; tau=0 mean {zero_mean:+.3f} n={len(zero_delta)}")
    TEXT, MUTED = "#0b0b0b", "#52514e"
    color_map = {"cellA": "#2a78d6", "cellB": "#eb6834"}   # validated categorical slots 1-2
    cell_label = {"cellA": "Cur-Pre", "cellB": "Raw-Post"}
    fig = plt.figure(figsize=(6.8, 3.2))
    # The tau = 0 strip is drawn only when such queries exist (with tau measured on the first
    # indexed non-test gold file there are none in Cur-Pre + Raw-Post).
    if len(zero_delta):
        gs = fig.add_gridspec(1, 7, wspace=0.12)
        ax_zero = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[0, 1:], sharey=ax_zero)
        axes = (ax_zero, ax_main)
    else:
        ax_zero, ax_main = None, fig.add_subplot(1, 1, 1)
        axes = (ax_main,)
        ax_main.set_ylabel(r"premium $\Delta$ (nDCG@10)", fontsize=9, color=TEXT)
    for ax in axes:
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED); ax.spines[side].set_linewidth(0.6)
        ax.tick_params(colors=MUTED, length=2.5, width=0.6, labelsize=8)
    # tau = 0 panel: queries sharing no token with their first gold file
    if ax_zero is not None:
        jitter = RNG.uniform(-0.3, 0.3, size=len(zero_delta))
        ax_zero.scatter(jitter, zero_delta, s=8, alpha=0.45, color="#9a9994", edgecolors="none", zorder=2)
        ax_zero.errorbar([0], [zero_mean], yerr=[[zero_mean - zero_lo], [zero_hi - zero_mean]],
                         fmt="o", color=TEXT, markersize=5, capsize=2.5, lw=1.2, zorder=5)
        ax_zero.set_xlim(-0.9, 0.9)
        ax_zero.set_xticks([0])
        ax_zero.set_xticklabels(["$\\tau=0$\n(no shared\ntoken)"], fontsize=7.5, color=TEXT)
        ax_zero.set_ylabel(r"premium $\Delta$ (nDCG@10)", fontsize=9, color=TEXT)
    # main panel: per-query scatter, quartile means with bootstrap CIs
    for c in ("cellA", "cellB"):
        mask = cells == c
        ax_main.scatter(x[mask], y[mask], s=8, alpha=0.35, color=color_map[c], edgecolors="none", zorder=2,
                        label=f"{cell_label[c]} queries (n={int(mask.sum())})")
    for i, (lo, hi) in enumerate(zip(bin_los, bin_his)):
        ax_main.add_patch(Rectangle((q_edges[i], lo), q_edges[i + 1] - q_edges[i], hi - lo,
                                    color="#52514e", alpha=0.14, lw=0, zorder=3))
    ax_main.errorbar(bin_centers, bin_means,
                     yerr=[[m - lo for m, lo in zip(bin_means, bin_los)], [hi - m for m, hi in zip(bin_means, bin_his)]],
                     fmt="o-", color=TEXT, markersize=5, lw=1.4, capsize=2.5, zorder=6,
                     label="quartile mean (95% CI)")
    ax_main.set_xlabel(r"query distinctiveness $\tau$ (max IDF of a query token shared with the gold file)",
                       fontsize=8.5, color=TEXT)
    ax_main.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7.5,
                   markerscale=1.4, labelcolor=TEXT, columnspacing=1.2, handletextpad=0.3)
    ax_main.text(0.01, 0.02, f"Pearson $r={pearson_r:+.3f}$ (permutation $p={perm_p:.3f}$)",
                 transform=ax_main.transAxes, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    plt.tight_layout()
    out_pdf = FIG_DIR / "fig_hero_phase_diagram.pdf"
    out_png = FIG_DIR / "fig_hero_phase_diagram.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[fig] wrote {out_pdf}")
    print(f"[stats] r={pearson_r:+.3f}, perm p={perm_p:.4f}, ρ={spearman_r:+.3f}")
    print(f"[stats] Q1→Q4: {' → '.join(f'{m:+.4f}' for m in bin_means)}")
    print(f"[stats] Q1−Q4 gap: {bin_means[0] - bin_means[-1]:+.4f}")
    print(f"[stats] τ=0 mean: {zero_mean:+.4f} (n={len(zero_delta)})")


if __name__ == "__main__":
    main()
