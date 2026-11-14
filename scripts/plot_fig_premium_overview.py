"""Figure 1 — the dense premium under three lexical baselines.

Two query sources, the ones the introduction contrasts: curated SWE-bench
Verified statements (Cell A) and raw GitHub PR descriptions (Cell B, the raw
cell whose +0.152 the text quotes; the two raw pre-cutoff vintage controls are
Q2 material and stay in tab:r8 / tab:genarm-tuned). For each, the canonical
encoder's nDCG@10 minus BM25 under three lexical conditions:
  BM25 at the common c=1000 chunking -> BM25 over whole files (its tuned size)
  -> whole-file BM25 on the query plus generated anchors (generate-then-grep).
Bars are means with repository-clustered 95% bootstrap CIs, read from
${REPO_PREMIUM_DATA:-data}/analysis/harmonized_overview.json (see the R = ... line).

Color: the three conditions are an ordered progression (more lexical help), so
one blue hue stepped light->dark (dataviz reference ramp, steps 250/450/650;
adjacent CVD dE 19.0, normal-vision dE 19.5). The palest step is below 3:1
against the surface, so every bar carries a thin outline; a direct value
label would be the usual relief; here Table 3 (tab:genarm-tuned) and tab:r8
carry the table view; with six bars, values are direct-labeled at the whisker.
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
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[1]
# Source-file labels. Cur-Pre from analysis/indexed/genarm_rebase.json; the raw pools are the harmonized
# pools (harmonized_analysis.py), assembled with Cur-Pre into analysis/harmonized_overview.json.
R = json.loads((DATA_ROOT / 'analysis/harmonized_overview.json').read_text())["cells"]
CELLS = [("A", "Cur-Pre"), ("B", "Raw-Post"), ("RA", "Raw-Pre'23"), ("RB", "Raw-Pre'22")]
CONDS = [("premium_c1000", "untuned BM25", "#86b6ef"),
         ("premium_tuned", "tuned BM25", "#2a78d6"),
         ("residual_tuned", "tuned + anchors", "#104281")]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

fig, ax = plt.subplots(figsize=(3.3, 2.5))
x = np.arange(len(CELLS))
w = 0.27
for j, (key, label, color) in enumerate(CONDS):
    m = np.array([R[c][key]["mean"] for c, _ in CELLS])
    lo = np.array([R[c][key]["ci95"][0] for c, _ in CELLS])
    hi = np.array([R[c][key]["ci95"][1] for c, _ in CELLS])
    xs = x + (j - 1) * w
    ax.bar(xs, m, width=w * 0.88, color=color, edgecolor="#0d366b", linewidth=0.4,
           label=label, zorder=2)
    ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="none", ecolor=MUTED, elinewidth=0.7,
                capsize=1.6, capthick=0.7, zorder=3)

ax.axhline(0, color=TEXT, linewidth=0.7, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([n for _, n in CELLS], fontsize=7.6)
ax.set_ylabel(r"dense premium ($\Delta$nDCG@10)")
ax.set_ylim(-0.17, 0.36)
ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
    ax.spines[side].set_linewidth(0.6)
ax.tick_params(colors=MUTED, length=2.5, width=0.6)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_color(TEXT)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 1.0), frameon=False, ncol=3,
           handlelength=0.9, handleheight=0.8, columnspacing=0.6, handletextpad=0.3,
           labelcolor=TEXT, title="dense premium against", title_fontsize=8)
fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.84))

out = ROOT / "figures/coreb_repo/fig_premium_overview"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{out}.pdf", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
print("wrote", f"{out}.pdf")
