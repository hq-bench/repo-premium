"""Fig 8 — query-type decomposition of the embedding premium (R7).

Stratifies all 611 queries by `n_repo_hits` (count of query-extracted identifiers
that resolve to a definition in the repo's def_index):
  - symbol-named  : ≥3 repo-resident identifiers  (69% pooled)
  - weakly-anchored: 1-2 hits                      (23%)
  - graph-unservable (a.k.a. descriptive): 0 hits  (7%)

For each bucket, plots pooled retriever nDCG@10 — showing that:
  - graph search collapses to 0 on the graph-unservable bucket,
  - BM25 also drops sharply there,
  - dense is the only working retriever on graph-unservable queries.

Reads:
  ${REPO_PREMIUM_DATA:-data}/r7_strat_pooled.jsonl  (produced by /tmp/r7_stratify.py)

Writes:
  figures/coreb_repo/fig8_bucket_decomposition.png
"""
from __future__ import annotations
from paths import DATA_ROOT

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

SRC = (DATA_ROOT / 'r7_strat_pooled.jsonl')
OUT = Path(__file__).resolve().parents[1] / "figures" / "coreb_repo"
OUT.mkdir(parents=True, exist_ok=True)

BUCKETS = [
    ("symbol-named (≥3 hits)", "#3a6ea5"),
    ("weakly-anchored (1-2 hits)", "#6fa86b"),
    ("descriptive (0 hits)", "#c8503d"),
]

RETRIEVERS = [
    ("grep", "grep", "#cccccc"),
    ("bm25", "BM25", "#999999"),
    ("graph", "graph (pure)", "#2ca02c"),
    ("dense_jina-1.5b", "dense (jina-code-1.5b)", "#1f77b4"),
    ("dense_Qwen3-4B", "dense (Qwen3-Emb-4B)", "#d62728"),
    ("rerank_jina-1.5b", "dense + graph rerank", "#dd8452"),
]


def hide(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load():
    return [json.loads(l) for l in open(SRC)]


def pooled_by_bucket(queries):
    buckets = defaultdict(list)
    for q in queries:
        buckets[q["_bucket"]].append(q)
    out = {}
    for label, _ in BUCKETS:
        items = buckets.get(label, [])
        if not items:
            continue
        d = {"n": len(items)}
        for key, _, _ in RETRIEVERS:
            vals = [q[key] for q in items if key in q]
            d[key] = float(np.mean(vals)) if vals else np.nan
        out[label] = d
    return out


def by_cell_descriptive(queries):
    out = {"A": [], "B": []}
    for q in queries:
        if q["_bucket"].startswith("descriptive"):
            out[q["_cell"]].append(q)
    res = {}
    for cell, items in out.items():
        d = {"n": len(items)}
        for key, _, _ in RETRIEVERS:
            vals = [q[key] for q in items if key in q]
            d[key] = float(np.mean(vals)) if vals else np.nan
        res[cell] = d
    return res


def plot_pooled(ax, summary):
    n_b = len(BUCKETS)
    n_r = len(RETRIEVERS)
    width = 0.8 / n_r
    x = np.arange(n_b)
    for i, (key, lab, color) in enumerate(RETRIEVERS):
        vals = []
        for b_label, _ in BUCKETS:
            s = summary.get(b_label, {})
            vals.append(s.get(key, 0))
        offset = (i - (n_r - 1) / 2) * width
        rects = ax.bar(x + offset, vals, width=width, color=color,
                       alpha=0.92, label=lab)
        for r, v in zip(rects, vals):
            if v > 0.01:
                ax.text(r.get_x() + r.get_width() / 2, v + 0.008,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6.5,
                        rotation=0)
    labels = []
    for b_label, _ in BUCKETS:
        n = summary.get(b_label, {}).get("n", 0)
        labels.append(f"{b_label}\n(n={n}, {n/611*100:.0f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("pooled nDCG@10")
    ax.set_ylim(0, 0.72)
    ax.set_title("Pooled (Cell A + B, n=611): which retriever serves which query type",
                 fontsize=10)
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=7.5)
    hide(ax)


def plot_descriptive_by_cell(ax, descr):
    cells = ["A", "B"]
    width = 0.8 / len(RETRIEVERS)
    x = np.arange(len(cells))
    for i, (key, lab, color) in enumerate(RETRIEVERS):
        vals = [descr[c].get(key, 0) for c in cells]
        offset = (i - (len(RETRIEVERS) - 1) / 2) * width
        rects = ax.bar(x + offset, vals, width=width, color=color,
                       alpha=0.92, label=lab)
        for r, v in zip(rects, vals):
            if v > 0.005:
                ax.text(r.get_x() + r.get_width() / 2, v + 0.008,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    labels = []
    for c in cells:
        n = descr[c]["n"]
        cell_full = "Cell A (curated)" if c == "A" else "Cell B (real PRs)"
        labels.append(f"{cell_full}\n(n={n})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("nDCG@10 (descriptive bucket only)")
    ax.set_ylim(0, 0.7)
    ax.set_title("Descriptive bucket alone — where dense is the only retriever that works",
                 fontsize=10)
    hide(ax)
    # No legend (same as top panel).


def main():
    queries = load()
    summary = pooled_by_bucket(queries)
    descr_by_cell = by_cell_descriptive(queries)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2),
                              gridspec_kw={"height_ratios": [1.4, 1]})
    plot_pooled(axes[0], summary)
    plot_descriptive_by_cell(axes[1], descr_by_cell)
    fig.suptitle("Fig 8 — Query-type decomposition: the embedding premium "
                  "lives in the graph-unservable slice",
                  fontsize=11, y=1.00)
    fig.tight_layout()
    out = OUT / "fig8_bucket_decomposition.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    print("\nPooled by bucket:")
    for k, v in summary.items():
        print(f"  {k}: n={v['n']}")
        for rkey, rlab, _ in RETRIEVERS:
            print(f"    {rlab:<26s} = {v.get(rkey, 0):.4f}")
    print("\nDescriptive by cell:")
    for c, v in descr_by_cell.items():
        print(f"  Cell {c}: n={v['n']}")
        for rkey, rlab, _ in RETRIEVERS:
            print(f"    {rlab:<26s} = {v.get(rkey, 0):.4f}")


if __name__ == "__main__":
    main()
