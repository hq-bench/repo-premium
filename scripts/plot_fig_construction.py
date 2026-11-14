"""Construction-pipeline figure for the ACL paper (fig:construction, Section 3).

Mirrors the six steps of the Section 3 text: repositories -> query mining
(curated SWE-bench Verified / raw GitHub PRs) -> gold labels -> quality filter
-> preprocessing -> corpus, yielding the four core pools. Counts match the
Step 4 paragraph (497 + 2,236 = 2,733 candidates, 1,265 retained). Fills tint
the source (blue = curated, orange = raw; light tints of palette slots 1-2),
shared steps are neutral; text is ink, never series color.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42})
ROOT = Path(__file__).resolve().parents[1]
TEXT, MUTED, EDGE = "#0b0b0b", "#52514e", "#8a8984"
CUR, RAW, SHARED, REPO = "#e1ecfa", "#fde6da", "#f1f1ef", "#f1f1ef"
CUR_E, RAW_E = "#2a78d6", "#eb6834"

fig, ax = plt.subplots(figsize=(7.0, 1.75))
ax.set_xlim(-0.012, 1.012); ax.set_ylim(0.0, 1.0); ax.axis("off")


def box(x, y, w, h, title, sub="", fc=SHARED, ec=EDGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.018",
                                fc=fc, ec=ec, lw=0.8))
    if sub:
        ax.text(x + w / 2, y + h * 0.76, title, ha="center", va="center", fontsize=7.5,
                weight="bold", color=TEXT)
        ax.text(x + w / 2, y + h * 0.36, sub, ha="center", va="center", fontsize=6.9,
                color=MUTED, linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=7.0, color=TEXT)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, mutation_scale=7,
                                shrinkA=0, shrinkB=0))


MID, H = 0.46, 0.52
cols = [0.000, 0.135, 0.320, 0.460, 0.600, 0.740]
W = [0.110, 0.160, 0.112, 0.112, 0.112, 0.112]
for i, x in enumerate(cols):
    ax.text(x + W[i] / 2, 0.975, f"Step {i + 1}", ha="center", va="center", fontsize=7.3, color=MUTED)

box(cols[0], MID - H / 2, W[0], H, "Repositories", "10 Python repos\n(SWE-bench\nVerified), pinned", REPO)
box(cols[1], 0.50, W[1], 0.41, "Curated mining", "SWE-bench Verified\nstatements (497)", CUR, CUR_E)
box(cols[1], 0.01, W[1], 0.41, "Raw mining", "merged-PR title + body,\n3 date windows (2,236)", RAW, RAW_E)
box(cols[2], MID - H / 2, W[2], H, "Gold labels", "source files the\nchange edited,\nbinary per file")
box(cols[3], MID - H / 2, W[3], H, "Quality filter", "1–3 gold files,\ncode changes\n2,733 → 1,265")
box(cols[4], MID - H / 2, W[4], H, "Preprocessing", "strip PR\nboilerplate\n(all queries)")
box(cols[5], MID - H / 2, W[5], H, "Corpus", "one doc per\n.py file, one\nsnapshot per repo")

outs = [("Cur-Pre  472", CUR, CUR_E), ("Raw-Post  288", RAW, RAW_E),
        ("Raw-Pre'23  138", RAW, RAW_E), ("Raw-Pre'22  367", RAW, RAW_E)]
ox, ow, oh = 0.882, 0.118, 0.19
oys = [0.81, 0.585, 0.355, 0.125]
for (t, fc, ec), y in zip(outs, oys):
    box(ox, y - oh / 2, ow, oh, t, fc=fc, ec=ec)

arrow(cols[0] + W[0], MID + 0.08, cols[1], 0.705)
arrow(cols[0] + W[0], MID - 0.08, cols[1], 0.215)
arrow(cols[1] + W[1], 0.705, cols[2], MID + 0.08)
arrow(cols[1] + W[1], 0.215, cols[2], MID - 0.08)
for i in range(2, 5):
    arrow(cols[i] + W[i], MID, cols[i + 1], MID)
for y in oys:
    arrow(cols[5] + W[5], MID, ox, y)

fig.tight_layout(pad=0.02)
out = ROOT / "figures/coreb_repo/fig_construction"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(f"{out}.png", dpi=220, bbox_inches="tight", pad_inches=0.01)
print("wrote", f"{out}.pdf")
