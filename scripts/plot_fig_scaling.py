"""Appendix figure (Q3): per-repository premium for jina-code-0.5b vs 1.5b.

Vector rebuild of the old fig5_encoder_scaling.png from the canonical caches
(${REPO_PREMIUM_DATA:-data}/smoke_test_results/canon202607/jina{05,15}_cell{A,B}_c1000).
Two panels (Cur-Pre, Raw-Post); per-repository mean premium for each size;
the pooled (query-weighted) means are given above each panel. The two sizes
are an ordered pair, so one blue hue light/dark (validated: CVD dE 19+).
"""
from __future__ import annotations
from paths import DATA_ROOT

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
ROOT = Path(__file__).resolve().parents[1]
CANON = DATA_ROOT / 'smoke_test_results/canon202607'
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SIZES = [("jina05", "jina-code-0.5b", "#86b6ef"), ("jina15", "jina-code-1.5b", "#104281")]
SHORT = {"scikit-learn__scikit-learn": "sklearn", "matplotlib__matplotlib": "matplotlib"}


def load(tag: str, cell: str) -> dict[str, list[float]]:
    out = defaultdict(list)
    for f in sorted((CANON / f"{tag}_cell{cell}_c1000").glob("*/smoke_test_django.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[f.parent.name].append(r["dense_ndcg10"] - r["bm25_ndcg10"])
    return out


def name(repo: str) -> str:
    return SHORT.get(repo, repo.split("__")[1])


fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6), gridspec_kw={"width_ratios": [10, 8], "wspace": 0.18})
for ax, (cell, title) in zip(axes, (("A", "Cur-Pre"), ("B", "Raw-Post"))):
    data = {tag: load(tag, cell) for tag, _, _ in SIZES}
    repos = sorted(set(data["jina05"]) & set(data["jina15"]), key=name)
    x = np.arange(len(repos))
    w = 0.38
    pooled = []
    for j, (tag, label, color) in enumerate(SIZES):
        vals = [np.mean(data[tag][r]) for r in repos]
        ax.bar(x + (j - 0.5) * w, vals, width=w * 0.9, color=color, edgecolor="#0d366b", linewidth=0.35,
               label=label, zorder=2)
        allq = [v for r in repos for v in data[tag][r]]
        pooled.append(np.mean(allq))
    n = sum(len(data["jina05"][r]) for r in repos)
    ax.set_title(f"{title} (n={n}): pooled $\\Delta$ {pooled[0]:+.3f} $\\rightarrow$ {pooled[1]:+.3f}",
                 fontsize=8, color=TEXT, loc="left")
    ax.axhline(0, color=TEXT, lw=0.6, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([name(r) for r in repos], rotation=35, ha="right", fontsize=7, color=TEXT)
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED); ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, length=2.5, width=0.6, labelsize=7)
axes[0].set_ylabel(r"premium $\Delta$ (nDCG@10)", color=TEXT)
axes[0].legend(loc="upper left", frameon=False, fontsize=7, labelcolor=TEXT)
lo = min(a.get_ylim()[0] for a in axes); hi = max(a.get_ylim()[1] for a in axes)
for a in axes:
    a.set_ylim(lo, hi)

out = ROOT / "figures/coreb_repo/fig5_encoder_scaling"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{out}.pdf", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=220, bbox_inches="tight")
print("wrote", f"{out}.pdf")
