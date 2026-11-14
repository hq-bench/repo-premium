"""Appendix figure: the premium-decay regularity across evaluation slices.

Rebuilt from the project data (no /tmp inputs) using the slice table and the
within-(slice, repository) tau percentile of law_clustered_bootstrap.py, so it
matches the slope analysis reported in the paper.

(a) per-slice quintile means of the premium on the within-repository tau
    percentile, centered on the slice mean (thin lines, colored by slice
    family), with the pooled mean and a slice-bootstrap 95% band;
(b) per-slice OLS slope of the premium on the percentile, 95% CI from
    resampling repositories within the slice (queries for the four
    single-repository language slices), readable slice names, and the
    common fixed-effects slope from law_clustered_bootstrap.json.

Family colors are slots 1-6 of the validated categorical palette, in order.
"""
from __future__ import annotations
from paths import DATA_ROOT

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from law_clustered_bootstrap import load_all, within_repo_obs  # noqa: E402

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
ROOT = Path(__file__).resolve().parents[1]
LAWJ = json.loads((DATA_ROOT / 'analysis/law_clustered_bootstrap.json').read_text())
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAMILIES = [  # (prefix test, label, color)
    ("py_cur_", "Cur-Pre (curated)", "#2a78d6"),
    ("py_raw_", "Raw-Post", "#eb6834"),
    ("py_r2", "Raw-Pre'22 / '23", "#1baf7a"),
    ("py_ood_", "Held-out pool", "#eda100"),
    ("py_desc_", "Desc (LLM-described)", "#e87ba4"),
    ("", "Other languages", "#6f4fb5"),
]
ENC = {"jina05": "jina-code-0.5b", "jina15": "jina-code-1.5b", "qwen06": "Qwen3-0.6B", "qwen4": "Qwen3-4B",
       "qwen8": "Qwen3-8B", "c2llm05": "C2LLM-0.5B", "c2llm7": "C2LLM-7B", "f2llm06": "F2LLM-0.6B",
       "f2llm4": "F2LLM-4B", "e5base": "E5-base", "e5large": "E5-large", "gteq15": "gte-Qwen2-1.5B",
       "gteq7": "gte-Qwen2-7B"}
LANG = {"ts": "TypeScript", "go": "Go", "rust": "Rust", "js": "JavaScript (vuejs)"}


def family(name: str) -> int:
    for i, (pre, _, _) in enumerate(FAMILIES):
        if pre and name.startswith(pre):
            return i
    return len(FAMILIES) - 1


def pretty(name: str) -> str:
    enc = ENC.get(name.rsplit("_", 1)[-1], name.rsplit("_", 1)[-1])
    if name.startswith("py_r22"):
        return f"Raw-Pre'22 · {enc}"
    if name.startswith("py_r23"):
        return f"Raw-Pre'23 · {enc}"
    short = {0: "Cur-Pre", 1: "Raw-Post", 3: "Held-out", 4: "Desc"}
    if name.startswith("py_"):
        return f"{short[family(name)]} · {enc}"
    return f"{LANG[name.split('_')[0]]} · {enc}"


def main() -> None:
    names = LAWJ["core_slices"]
    slices = load_all()
    names = [n for n in names if n in slices]
    obs = within_repo_obs(slices, names)
    rng = np.random.default_rng(20260704)
    edges = np.linspace(0, 1, 6)
    centers = (edges[:-1] + edges[1:]) / 2
    curves, slopes = {}, {}
    for si, name in enumerate(names):
        m = obs["s"] == si
        x, y, repo = obs["x"][m], obs["y"][m], obs["repo"][m]
        b = np.clip(np.digitize(x, edges[1:-1]), 0, 4)
        curves[name] = np.array([y[b == k].mean() if (b == k).any() else np.nan for k in range(5)]) - y.mean()
        est = np.polyfit(x, y, 1)[0]
        repos = np.unique(repo)
        idx = {r: np.flatnonzero(repo == r) for r in repos}
        boots = []
        for _ in range(2000):
            if len(repos) > 1:   # resample repositories within the slice
                ix = np.concatenate([idx[r] for r in rng.choice(repos, len(repos))])
            else:                # single-repository slice: resample queries
                ix = rng.integers(0, len(x), len(x))
            if np.ptp(x[ix]) > 0:
                boots.append(np.polyfit(x[ix], y[ix], 1)[0])
        slopes[name] = (est, np.percentile(boots, 2.5), np.percentile(boots, 97.5))
    C = np.array([curves[n] for n in names])
    pooled = np.nanmean(C, axis=0)
    band = np.array([np.nanmean(C[rng.integers(0, len(names), len(names))], axis=0) for _ in range(4000)])
    lo, hi = np.percentile(band, [2.5, 97.5], axis=0)

    fig = plt.figure(figsize=(6.4, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.62)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for ax in (ax_a, ax_b):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED); ax.spines[side].set_linewidth(0.6)
        ax.tick_params(colors=MUTED, length=2.5, width=0.6, labelsize=8)

    for name in names:
        ax_a.plot(centers, curves[name], color=FAMILIES[family(name)][2], lw=0.7, alpha=0.45, zorder=2)
    ax_a.fill_between(centers, lo, hi, color=MUTED, alpha=0.18, lw=0, zorder=3)
    ax_a.plot(centers, pooled, color=TEXT, lw=1.8, marker="o", ms=4, zorder=4, label="pooled mean (95% band)")
    ax_a.axhline(0, color=MUTED, lw=0.6, zorder=1)
    ax_a.set_xlabel(r"within-repository percentile of $\tau$", color=TEXT)
    ax_a.set_ylabel(r"premium $\Delta$, slice-centered (quintile mean)", color=TEXT)
    ax_a.set_title("(a) Decay shape across slices", fontsize=8.5, color=TEXT, loc="left")
    ax_a.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    handles = [plt.Line2D([], [], color=c, lw=1.6, label=l) for _, l, c in FAMILIES]
    handles.append(plt.Line2D([], [], color=TEXT, lw=1.8, marker="o", ms=4, label="pooled mean (95% band)"))
    ax_a.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.14), ncol=2, frameon=False, fontsize=7.5,
                labelcolor=TEXT, handlelength=1.4, columnspacing=1.0)

    order = sorted(names, key=lambda n: (family(n), slopes[n][0]))
    ys = np.arange(len(order))[::-1]
    for yv, name in zip(ys, order):
        est, l, h = slopes[name]
        col = FAMILIES[family(name)][2]
        ax_b.plot([l, h], [yv, yv], color=col, lw=1.0, zorder=2)
        ax_b.plot([est], [yv], "o", color=col, ms=3.2, zorder=3)
    common = LAWJ["common_slope"]
    ax_b.axvline(0, color=MUTED, lw=0.6, zorder=1)
    ax_b.axvline(common, color=TEXT, lw=0.9, ls="--", zorder=1)
    ax_b.text(common, len(order) - 0.2, f" common slope {common:+.3f}", fontsize=7.5, color=TEXT, va="bottom", ha="left")
    ax_b.set_yticks(ys)
    ax_b.set_yticklabels([pretty(n) for n in order], fontsize=7, color=TEXT)
    ax_b.set_ylim(-0.8, len(order) + 0.6)
    ax_b.set_xlabel(r"slope of $\Delta$ on the $\tau$ percentile (95% CI)", color=TEXT)
    ax_b.set_title("(b) Per-slice slopes", fontsize=8.5, color=TEXT, loc="left")
    ax_b.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)

    out = ROOT / "figures/coreb_repo/fig_law_collapse"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=220, bbox_inches="tight")
    neg = sum(slopes[n][0] < 0 for n in names)
    print(f"slices {len(names)}, negative slopes {neg}, common {common:+.3f}")


if __name__ == "__main__":
    main()
