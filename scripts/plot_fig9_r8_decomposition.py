"""Fig 9 — Q2 decomposition: curated pre-cutoff vs raw pre-cutoff vs raw post-cutoff.

Three bars (pooled per-query premium, canonical jina-code-0.5b at c=1000,
stripped queries) with the same paired query-level 95% bootstrap CIs as
tab:r8, plus the decomposition of the Raw-Post minus Cur-Pre gap into a
mining step (Raw-Pre'23 minus Cur-Pre) and a date step (Raw-Post minus
Raw-Pre'23). Repository-block intervals for these contrasts include zero;
the caption says so, and the figure carries no claim beyond the point
estimates and query-level CIs.

Reads the canonical per-query caches directly (no /tmp intermediates) and
checks the three means against tab:r8 before plotting.

Writes figures/coreb_repo/fig9_r8_decomposition.{pdf,png} and the copy used
by the ACL submission.
"""
from __future__ import annotations
from paths import DATA_ROOT

import json
from pathlib import Path
from random import Random

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

ROOT = Path(__file__).resolve().parents[1]
SMOKE = DATA_ROOT / "smoke_test_results"
CELLS = [  # label, cache dir, tab:r8 published mean
    ("Cur-Pre\ncurated, pre-cutoff", SMOKE / "canon202607" / "jina05_cellA_c1000", 0.121),
    ("Raw-Pre'23\nraw, pre-cutoff", SMOKE / "r8_robust_jina-0.5b_c1000", 0.205),
    ("Raw-Post\nraw, post-cutoff", SMOKE / "canon202607" / "jina05_cellB_c1000", 0.204),
]
OUT = ROOT / "figures" / "coreb_repo"


def deltas(d: Path) -> list[float]:
    out = []
    for f in sorted(d.glob("*/smoke_test_django.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out.append(r["dense_ndcg10"] - r["bm25_ndcg10"])
    return out


def boot_ci(xs: list[float], n_boot: int = 10000, seed: int = 0):
    rng = Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(xs) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


stats = []
for label, d, published in CELLS:
    xs = deltas(d)
    m, lo, hi = boot_ci(xs)
    assert abs(round(m, 3) - published) < 1.5e-3, (label, m, published)
    stats.append((label, len(xs), m, lo, hi))
    print(f"{label.splitlines()[0]:11s} n={len(xs):3d}  delta={m:+.4f}  CI=[{lo:+.3f}, {hi:+.3f}]")

(_, _, a, _, _), (_, _, ra, _, _), (_, _, b, _, _) = stats
print(f"gap {b - a:+.3f} = mining {ra - a:+.3f} + date {b - ra:+.3f}")

fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.arange(3)
means = [s[2] for s in stats]
err = [[s[2] - s[3] for s in stats], [s[4] - s[2] for s in stats]]
colors = ["#3a6ea5", "#d4a02f", "#c8503d"]
bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.6, width=0.55,
              yerr=err, capsize=6, ecolor="#333333", error_kw={"lw": 1.0})
for i, bar in enumerate(bars):
    ax.text(bar.get_x() + bar.get_width() / 2, means[i] + err[1][i] + 0.01,
            f"{means[i]:+.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

top = max(s[4] for s in stats)
y1, y2 = top + 0.10, top + 0.045
ax.annotate("", xy=(2.0, y1), xytext=(0.0, y1), arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.0))
ax.text(1.0, y1 + 0.006, f"gap {b - a:+.3f}", ha="center", va="bottom", fontsize=9)
ax.annotate("", xy=(1.0, y2), xytext=(0.0, y2), arrowprops=dict(arrowstyle="<->", color="#1f6f43", lw=1.0))
ax.text(0.5, y2 + 0.006, f"mining {ra - a:+.3f}", ha="center", va="bottom", fontsize=8, color="#1f6f43")
ax.annotate("", xy=(2.0, y2), xytext=(1.0, y2), arrowprops=dict(arrowstyle="<->", color="#aa5511", lw=1.0))
ax.text(1.5, y2 + 0.006, f"date {b - ra:+.3f}", ha="center", va="bottom", fontsize=8, color="#aa5511")

ax.set_xticks(x)
ax.set_xticklabels([f"{s[0]}\nn={s[1]}" for s in stats], fontsize=9)
ax.set_ylabel(r"$\Delta$ = nDCG@10(dense) $-$ nDCG@10(BM25)")
ax.set_ylim(0, y1 + 0.05)
ax.axhline(0, color="#888888", lw=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25, lw=0.5)
plt.tight_layout()

OUT.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT / "fig9_r8_decomposition.png", dpi=180, bbox_inches="tight")
plt.savefig(OUT / "fig9_r8_decomposition.pdf", bbox_inches="tight")
print(f"wrote {OUT / 'fig9_r8_decomposition.pdf'}")
