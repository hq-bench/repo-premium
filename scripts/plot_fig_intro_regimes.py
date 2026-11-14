"""Introduction figures — exemplar queries and the decay curve.

Figure: fig_intro_queries.{pdf,png}
  Three exemplar queries from the released caches, one per retrieval
  regime, with their actual per-query BM25 and dense nDCG@10.

Figure: fig_intro_decay.{pdf,png}
  Decile means of the pooled premium over tau_MaxIDF with 95% CIs and
  a linear trend, tight y-axis; exemplar tau positions marked on the
  top edge. (Per-query scatter lives in the hero figure.)

Reads ${REPO_PREMIUM_DATA:-data}/coreb_pull (canonical jina05 cells + tau variants).

Usage:
  conda run -n llm python scripts/plot_fig_intro_regimes.py
"""
from __future__ import annotations
from paths import DATA_ROOT

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

# fig_intro_decay is placed in a 0.52\textwidth wrapfigure (~2.9in), so the
# 5.4in canvas is scaled ~0.53x in the paper; sizes here target ~8-9pt final.
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.labelsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

PULL = (DATA_ROOT / 'smoke_test_results')
FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "coreb_repo"
RNG = np.random.default_rng(0)

C_BM25 = "#8c8c8c"
C_DENSE = "#d95f02"
C_CURVE = "#d62728"

EXEMPLARS = [
    {"qid": "django__django-11728", "cell": "cellA", "letter": "A",
     "regime": "identifier-anchored (lexical regime)",
     "snippet": "simplify_regexp() doesn't replace\ntrailing groups"},
    {"qid": "pydata__xarray-6992", "cell": "cellA", "letter": "B",
     "regime": "mixed (hybrid regime)",
     "snippet": "index refactor: more _coord_names\nthan _variables on Dataset"},
    {"qid": "django__django-pr-21426", "cell": "cellB", "letter": "C",
     "regime": "behavior-described (semantic regime)",
     "snippet": "Skipped DB cache deletion when\nculling offset is zero"},
]


def load_pool():
    rows = {}
    for cell in ("cellA", "cellB"):
        for repo_dir in sorted((PULL / cell).iterdir()):
            f = repo_dir / "smoke_test_django.jsonl"
            if not f.exists():
                continue
            recs = {json.loads(l)["query_id"]: json.loads(l)
                    for l in f.read_text().splitlines() if l.strip()}
            tvf = PULL / "tau_variants" / cell / repo_dir.name / "tau_variants.jsonl"
            for line in tvf.read_text().splitlines():
                tv = json.loads(line)
                r = recs.get(tv["query_id"])
                if r:
                    rows[tv["query_id"]] = {
                        "tau": tv["tau_top1"],
                        "bm25": r["bm25_ndcg10"],
                        "dense": r["dense_ndcg10"],
                    }
    return rows


def fig_queries(pool) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ys = np.arange(len(EXEMPLARS))[::-1] * 1.0
    bh = 0.16
    for e, y in zip(EXEMPLARS, ys):
        r = pool[e["qid"]]
        ax.barh(y + bh / 2 + 0.02, r["bm25"], height=bh, color=C_BM25,
                label="BM25" if e["letter"] == "A" else None)
        ax.barh(y - bh / 2 - 0.02, r["dense"], height=bh, color=C_DENSE,
                label="dense" if e["letter"] == "A" else None)
        ax.text(-0.03, y + 0.30,
                f"({e['letter']})  “{e['snippet']}”",
                ha="left", va="bottom", fontsize=8.0, style="italic",
                transform=ax.get_yaxis_transform())
        ax.text(-0.03, y + 0.26, e["regime"] + rf"   $\tau={r['tau']:.1f}$",
                ha="left", va="top", fontsize=7.4, color="#444444",
                transform=ax.get_yaxis_transform())
        for v, yy in ((r["bm25"], y + bh / 2 + 0.02),
                      (r["dense"], y - bh / 2 - 0.02)):
            ax.text(v + 0.02, yy, f"{v:.2f}", va="center", fontsize=7.2)
    ax.set_xlim(0, 1.12)
    ax.set_ylim(-0.55, ys[0] + 0.75)
    ax.set_yticks([])
    ax.set_xlabel("per-query nDCG@10")
    ax.legend(loc="lower right", frameon=False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_intro_queries.{ext}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {FIG_DIR / 'fig_intro_queries.pdf'}")


def fig_decay(pool) -> None:
    tau = np.array([v["tau"] for v in pool.values()])
    dlt = np.array([v["dense"] - v["bm25"] for v in pool.values()])
    pos = tau > 0
    x, y = tau[pos], dlt[pos]

    edges = np.quantile(x, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, 9)
    cx, cy, lo, hi = [], [], [], []
    for i in range(10):
        m = idx == i
        vals = y[m]
        cx.append(float(x[m].mean()))
        cy.append(float(vals.mean()))
        bs = [vals[RNG.integers(0, len(vals), len(vals))].mean()
              for _ in range(2000)]
        lo.append(np.percentile(bs, 2.5))
        hi.append(np.percentile(bs, 97.5))
    cx, cy = np.array(cx), np.array(cy)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    xmin, xmax = float(x.min()) - 0.25, float(x.max()) + 0.25

    t1, t2 = np.quantile(x, [1 / 3, 2 / 3])

    # linear trend
    b1, b0 = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, b0 + b1 * xx, color="#1f77b4", lw=1.8, ls="--", alpha=0.85,
            label="linear fit", zorder=4)
    # decile means with soft CI band
    ax.fill_between(cx, lo, hi, color=C_CURVE, alpha=0.13, lw=0, zorder=3)
    ax.plot(cx, cy, "o-", color=C_CURVE, lw=2.4, markersize=7,
            zorder=5, label="decile mean")

    # exemplar positions, letters near the bottom
    for e in EXEMPLARS:
        t = pool[e["qid"]]["tau"]
        ax.axvline(t, color="#2ca02c", lw=1.1, ls=":", alpha=0.65, zorder=2)
        ax.annotate(f"({e['letter']})", (t, 0.03),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=14,
                    fontweight="bold", color="#2ca02c")

    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.4)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(min(-0.03, float(np.min(lo)) - 0.02), max(0.45, float(np.max(hi)) + 0.02))
    ax.set_xlabel(r"query distinctiveness $\tau$ (MaxIDF)")
    ax.set_ylabel(r"premium $\Delta$ (nDCG@10)")
    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 0.90))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_intro_decay.{ext}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {FIG_DIR / 'fig_intro_decay.pdf'}")
    print(f"  terciles at {t1:.2f}, {t2:.2f}; slope {b1:+.4f}")


def main() -> None:
    pool = load_pool()
    fig_queries(pool)
    fig_decay(pool)
    for e in EXEMPLARS:
        r = pool[e["qid"]]
        print(f"  ({e['letter']}) {e['qid']}: tau={r['tau']:.1f} "
              f"bm25={r['bm25']:.2f} dense={r['dense']:.2f}")


if __name__ == "__main__":
    main()
