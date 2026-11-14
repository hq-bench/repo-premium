"""Offline RG1 defenses: permutation test, variance decomposition, QPP-baseline table.

Reads:
  ${REPO_PREMIUM_DATA:-data}/coreb_pull/{cellA,cellB}/{repo}/smoke_test_django.jsonl  (per-query Δ)
  ${REPO_PREMIUM_DATA:-data}/coreb_pull/tau_variants/{cellA,cellB}/{repo}/tau_variants.jsonl  (4 τ variants)

Writes:
  ${REPO_PREMIUM_DATA:-data}/coreb_pull/rg1_stats.json  (all numbers for paper edits)
  figures/coreb_repo/fig_rg1_variance_decomp.pdf  (variance ceiling figure)

Usage:
  conda run -n llm python scripts/rg1_defenses.py
"""
from __future__ import annotations
from paths import DATA_ROOT
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
})

PULL = (DATA_ROOT / 'smoke_test_results')
FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "coreb_repo"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STATS_OUT = PULL / "rg1_stats.json"

RNG = np.random.default_rng(20260704)


def load_cell(cell: str) -> list[dict]:
    """Merge smoke_test_django.jsonl + tau_variants.jsonl by query_id."""
    delta_dir = PULL / cell
    tau_dir = PULL / "tau_variants" / cell
    rows = []
    for repo_dir in sorted(delta_dir.iterdir()):
        repo = repo_dir.name
        delta_map = {}
        with (repo_dir / "smoke_test_django.jsonl").open() as fh:
            for line in fh:
                r = json.loads(line)
                delta_map[r["query_id"]] = r
        tau_map = {}
        tv_file = tau_dir / repo / "tau_variants.jsonl"
        if tv_file.exists():
            with tv_file.open() as fh:
                for line in fh:
                    r = json.loads(line)
                    tau_map[r["query_id"]] = r
        for qid, d in delta_map.items():
            merged = {
                "query_id": qid,
                "repo": repo,
                "cell": cell,
                "delta": d["delta_ndcg10"],
                "bm25_ndcg10": d["bm25_ndcg10"],
                "dense_ndcg10": d["dense_ndcg10"],
                "tau_mean5": d.get("tau_q", 0.0),  # legacy field == tau_mean5
                "n_gold": len(d.get("gold_files", [])),
            }
            tv = tau_map.get(qid, {})
            for k in ("tau_top1", "tau_mean_all", "tau_jaccard", "n_q_tokens", "n_gold_tokens", "n_shared"):
                if k in tv:
                    merged[k] = tv[k]
            rows.append(merged)
    return rows


def quartile_means(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[int]]:
    """Return (quartile means of y, quartile sizes) using quantiles of x."""
    q_edges = np.quantile(x, [0, 0.25, 0.5, 0.75, 1.0])
    bin_idx = np.clip(np.digitize(x, q_edges[1:-1]), 0, 3)
    means = [float(y[bin_idx == i].mean()) if (bin_idx == i).any() else float("nan") for i in range(4)]
    sizes = [int((bin_idx == i).sum()) for i in range(4)]
    return means, sizes


def is_monotone_decreasing(vals: list[float]) -> bool:
    return all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def permutation_test(x: np.ndarray, y: np.ndarray, n_perm: int = 10_000) -> dict:
    """Two-sided permutation p-value for Pearson r and quartile-monotonicity indicator."""
    obs_r, obs_p_param = stats.pearsonr(x, y)
    obs_rho, _ = stats.spearmanr(x, y)
    obs_q, _ = quartile_means(x, y)
    obs_range = obs_q[0] - obs_q[3]  # Q1 - Q4 gap
    obs_mono = is_monotone_decreasing(obs_q)

    perm_r = np.empty(n_perm)
    perm_range = np.empty(n_perm)
    perm_mono = 0
    for i in range(n_perm):
        y_shuf = RNG.permutation(y)
        perm_r[i] = stats.pearsonr(x, y_shuf)[0]
        pq, _ = quartile_means(x, y_shuf)
        perm_range[i] = pq[0] - pq[3]
        if is_monotone_decreasing(pq):
            perm_mono += 1

    return {
        "n": int(len(x)),
        "pearson_r": float(obs_r),
        "pearson_p_parametric": float(obs_p_param),
        "pearson_p_permutation_2s": float((np.abs(perm_r) >= abs(obs_r)).mean()),
        "spearman_rho": float(obs_rho),
        "quartile_means": obs_q,
        "quartile_range_Q1_minus_Q4": float(obs_range),
        "quartile_range_p_permutation": float((perm_range >= obs_range).mean()),  # one-sided (larger Q1-Q4)
        "monotone_decreasing_observed": bool(obs_mono),
        "p_random_permutation_monotone": float(perm_mono / n_perm),
        "n_perm": n_perm,
    }


def variance_decomposition(rows: list[dict]) -> dict:
    """Total Δ variance and per-source decomposition:
      total = between-repo + within-repo
      total = between-quartile + within-quartile
      total = between-cell + within-cell
    Also report intrinsic ceiling estimate: how much Δ variance a single rank-swap induces.
    """
    delta = np.array([r["delta"] for r in rows])
    repos = np.array([r["repo"] for r in rows])
    cells = np.array([r["cell"] for r in rows])
    tau_jacc = np.array([r.get("tau_jaccard", 0.0) for r in rows])

    total = float(delta.var(ddof=0))

    def eta2(groups):
        gmean = delta.mean()
        num = 0.0
        for g in set(groups):
            mask = (groups == g)
            if mask.sum() < 2:
                continue
            num += mask.sum() * (delta[mask].mean() - gmean) ** 2
        return num / (len(delta) * total)

    eta2_repo = eta2(repos)
    eta2_cell = eta2(cells)

    # Quartile of τ_jaccard
    q_edges = np.quantile(tau_jacc, [0, 0.25, 0.5, 0.75, 1.0])
    bins = np.clip(np.digitize(tau_jacc, q_edges[1:-1]), 0, 3)
    eta2_qtau = eta2(bins)

    # Intrinsic ceiling: for a single gold-file query (typical case), a single rank swap in
    # top-10 changes nDCG@10 by up to (1/log2(2)) - (1/log2(3)) = 1 - 0.631 = 0.369 (rank 1↔2).
    # For rank 2↔3 the swing is 1/log2(3) - 1/log2(4) = 0.131. Average over rank 1..10 pairs:
    def ndcg_at_rank(r):
        return 1.0 / np.log2(r + 1)

    swap_swings = [abs(ndcg_at_rank(k) - ndcg_at_rank(k + 1)) for k in range(1, 10)]
    mean_swap_swing = float(np.mean(swap_swings))
    # Δ = dense - BM25, each single-swap induces up to ±swap on either side; upper bound on
    # variance contribution from independent 1-rank-swap noise on each retriever ≈ 2 * mean_swap^2 / 12
    # (uniform bounded RV). Report as a soft ceiling.
    approx_intrinsic_variance = (2.0 * mean_swap_swing ** 2) / 12.0

    return {
        "n_queries": int(len(delta)),
        "total_variance": total,
        "eta2_repo_on_delta": float(eta2_repo),
        "eta2_cell_on_delta": float(eta2_cell),
        "eta2_tau_quartile_on_delta": float(eta2_qtau),
        "r2_from_pearson_r": float(stats.pearsonr(tau_jacc, delta)[0] ** 2),
        "mean_1rank_ndcg_swap_swing": mean_swap_swing,
        "approx_intrinsic_variance_lower_bound": approx_intrinsic_variance,
        "approx_intrinsic_variance_as_frac_of_total": float(approx_intrinsic_variance / total),
    }


def qpp_baselines_r3_style(rows: list[dict]) -> dict:
    """QPP-flavoured predictor comparison over pooled τ>0 rows.

    Baselines:
      - query_length (n_q_tokens, higher = longer, expected sign +?)
      - max_idf (τ_top1, Hauff's MaxIDF)
      - avg_idf (τ_mean_all, Hauff's AvgIDF)
      - mean_idf_top5 (τ_mean5)
      - idf_jaccard (τ_jaccard, ours)
      - n_gold (multi-file scope confound)
    Report per predictor: Pearson r, Spearman ρ, monotone quartile flag,
    permutation p on both r and quartile monotonicity.
    """
    # keep τ_jaccard > 0 population (the paper's headline regime)
    rows = [r for r in rows if r.get("tau_jaccard", 0.0) > 0]
    delta = np.array([r["delta"] for r in rows])
    predictors = {
        "query_length_tokens": np.array([r.get("n_q_tokens", 0) for r in rows], dtype=float),
        "n_gold_files": np.array([r.get("n_gold", 1) for r in rows], dtype=float),
        "max_idf__tau_top1__Hauff_MaxIDF": np.array([r.get("tau_top1", 0.0) for r in rows]),
        "avg_idf__tau_mean_all__Hauff_AvgIDF": np.array([r.get("tau_mean_all", 0.0) for r in rows]),
        "mean_idf_top5__tau_mean5__ours_R3": np.array([r.get("tau_mean5", 0.0) for r in rows]),
        "idf_jaccard__tau_jaccard__ours_headline": np.array([r.get("tau_jaccard", 0.0) for r in rows]),
    }
    out = {"n_pop": int(len(rows))}
    for name, x in predictors.items():
        pt = permutation_test(x, delta, n_perm=10_000)
        out[name] = {
            "pearson_r": pt["pearson_r"],
            "pearson_p_perm": pt["pearson_p_permutation_2s"],
            "spearman_rho": pt["spearman_rho"],
            "quartile_means": pt["quartile_means"],
            "quartile_range_Q1_minus_Q4": pt["quartile_range_Q1_minus_Q4"],
            "quartile_range_p_perm": pt["quartile_range_p_permutation"],
            "monotone_decreasing": pt["monotone_decreasing_observed"],
            "p_random_permutation_monotone": pt["p_random_permutation_monotone"],
        }
    return out


def fig_variance_decomposition(varstats: dict, qpp: dict) -> Path:
    """Two-panel figure:
      (a) variance decomposition of Δ (stacked bar: between-repo / within-repo, τ-quartile / rest, intrinsic ceiling).
      (b) predictor-comparison forest plot: |r| ± bootstrap 95% CI for each QPP baseline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))

    # Panel A: variance components
    ax = axes[0]
    labels = [
        r"$\eta^2_{\mathrm{repo}}$",
        r"$\eta^2_{\mathrm{cell}}$",
        r"$\eta^2_{\tau\mathrm{-quartile}}$",
        r"$r^2_{\tau_{\mathrm{jaccard}}}$",
        r"intrinsic ceiling",
    ]
    values = [
        varstats["eta2_repo_on_delta"],
        varstats["eta2_cell_on_delta"],
        varstats["eta2_tau_quartile_on_delta"],
        varstats["r2_from_pearson_r"],
        varstats["approx_intrinsic_variance_as_frac_of_total"],
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"]
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="black", lw=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(r"fraction of Var($\Delta$) explained")
    ax.set_title(r"(a) $\Delta$-variance decomposition (pooled, $n=%d$)" % varstats["n_queries"], fontsize=9)
    ax.axhline(0, color="black", lw=0.5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, max(values) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: QPP predictor forest (|Pearson r| ± p-value annotation)
    ax = axes[1]
    keys = [
        ("query_length_tokens", "query length"),
        ("n_gold_files", "$n_{\\mathrm{gold}}$"),
        ("max_idf__tau_top1__Hauff_MaxIDF", "MaxIDF (Hauff)"),
        ("avg_idf__tau_mean_all__Hauff_AvgIDF", "AvgIDF (Hauff)"),
        ("mean_idf_top5__tau_mean5__ours_R3", r"$\tau_{\mathrm{mean5}}$"),
        ("idf_jaccard__tau_jaccard__ours_headline", r"$\tau_{\mathrm{jaccard}}$ (ours)"),
    ]
    rs = [qpp[k]["pearson_r"] for k, _ in keys]
    labels_b = [lbl for _, lbl in keys]
    y_pos = np.arange(len(rs))
    colors_b = ["#7f7f7f" if not qpp[k]["monotone_decreasing"] else "#2ca02c" for k, _ in keys]
    # highlight our headline
    colors_b[-1] = "#d62728"
    ax.barh(y_pos, rs, color=colors_b, edgecolor="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_b, fontsize=8.5)
    ax.set_xlabel(r"Pearson $r(\mathrm{predictor},\ \Delta)$")
    ax.set_title(r"(b) QPP predictor comparison, $\tau_{\mathrm{jaccard}}>0$, $n=%d$" % qpp["n_pop"], fontsize=9)
    # Annotate with permutation p and monotone flag
    for i, (k, _) in enumerate(keys):
        p_r = qpp[k]["pearson_p_perm"]
        mono = "mono" if qpp[k]["monotone_decreasing"] else "non-mono"
        ax.text(rs[i] + (0.005 if rs[i] >= 0 else -0.005), i,
                f"p={p_r:.3f} · {mono}",
                va="center", ha="left" if rs[i] >= 0 else "right",
                fontsize=7.2, color="black")
    ax.set_xlim(min(rs) - 0.10, max(0.02, max(rs) + 0.05))
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "fig_rg1_variance_decomp.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig_rg1_variance_decomp.png", dpi=200, bbox_inches="tight")
    plt.close()
    return out


def main():
    cellA = load_cell("cellA")
    cellB = load_cell("cellB")
    pooled = cellA + cellB

    # Restrict to headline population: τ_jaccard > 0
    pooled_tj = [r for r in pooled if r.get("tau_jaccard", 0.0) > 0]

    print(f"[load] cellA={len(cellA)}, cellB={len(cellB)}, pooled={len(pooled)}, pooled τ_j>0={len(pooled_tj)}")

    # === Headline permutation test on τ_jaccard ===
    x = np.array([r["tau_jaccard"] for r in pooled_tj])
    y = np.array([r["delta"] for r in pooled_tj])
    perm = permutation_test(x, y, n_perm=10_000)
    print("\n=== Permutation test (τ_jaccard, pooled τ>0) ===")
    print(json.dumps(perm, indent=2))

    # === Variance decomposition ===
    varstats = variance_decomposition(pooled_tj)
    print("\n=== Variance decomposition ===")
    print(json.dumps(varstats, indent=2))

    # === QPP baseline comparison ===
    qpp = qpp_baselines_r3_style(pooled)
    print("\n=== QPP predictor comparison (pooled τ_jaccard>0) ===")
    for k, v in qpp.items():
        if k == "n_pop":
            print(f"n_pop = {v}")
            continue
        print(f"  {k}: r={v['pearson_r']:+.3f} (p_perm={v['pearson_p_perm']:.3f}), "
              f"ρ={v['spearman_rho']:+.3f}, Q1-Q4={v['quartile_range_Q1_minus_Q4']:+.3f}, "
              f"mono={v['monotone_decreasing']}")

    # === Figure ===
    fig_path = fig_variance_decomposition(varstats, qpp)
    print(f"\n[fig] wrote {fig_path}")

    STATS_OUT.write_text(json.dumps({
        "permutation_test_headline": perm,
        "variance_decomposition": varstats,
        "qpp_baselines": qpp,
    }, indent=2))
    print(f"[stats] wrote {STATS_OUT}")


if __name__ == "__main__":
    main()
