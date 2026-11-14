"""Law deep-dive: pooled-slope significance, r22/r23 exception forensics,
and the collapse figure (candidate new hero plot).

Builds on law_mining.py's slice table. Per tau variant:
  1. pooled F-test: slice fixed effects only vs FE + common slope
     (is the shared tau-structure significant at all?), plus a
     slice-cluster bootstrap CI on the common slope
  2. r22/r23 forensics: per-repo slopes, quartile means, tau=0 mass
  3. collapse figure: per-slice quintile bin means, slice-centered,
     against within-slice tau percentile, with the pooled curve;
     panel (b) slope forest across all slices

Writes ${REPO_PREMIUM_DATA:-data}/coreb_law/law_deepdive.json and
figures/coreb_repo/fig_law_collapse.{pdf,png}.

Usage:
  conda run -n llm python scripts/law_deepdive.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from law_mining import (  # noqa: E402
    LAW, VARIANTS, ANCHOR, bin_means, load_slice, pct_rank, read_jsonl,
    slice_table, slope_ci,
)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "coreb_repo"
RNG = np.random.default_rng(0)
N_BOOT = 2000

# chunk-size slices reuse py_cur_jina05 queries -> pseudo-replicates
CHUNK_DUPES = {"py_cur_jina05_c1500", "py_cur_jina05_c2000", "py_cur_jina05_c3000"}

FAMILY = [
    ("curated", ("py_cur_",)),
    ("raw", ("py_raw_",)),
    ("raw pre-cutoff", ("py_r23_", "py_r22_")),
    ("desc", ("py_desc_",)),
    ("ood", ("py_ood_",)),
    ("language", ("js_", "ts_", "go_", "rust_")),
]
FAMILY_COLOR = {"curated": "#1f77b4", "raw": "#ff7f0e",
                "raw pre-cutoff": "#8c564b", "desc": "#9467bd",
                "ood": "#2ca02c", "language": "#d62728",
                "chunk": "#7f7f7f"}


def slice_family(name: str) -> str:
    if name in CHUNK_DUPES:
        return "chunk"
    for fam, prefixes in FAMILY:
        if any(name.startswith(p) for p in prefixes):
            return fam
    return "curated"


def load_all() -> dict[str, list[dict]]:
    slices = {}
    for name, (rdir, tdir) in slice_table().items():
        if not rdir.exists() or not tdir.exists():
            continue
        rows = load_slice(rdir, tdir)
        if len(rows) >= 20:
            slices[name] = rows
    return slices


def xy(rows: list[dict], variant: str) -> tuple[np.ndarray, np.ndarray]:
    tau = np.array([r[variant] for r in rows])
    dlt = np.array([r["delta"] for r in rows])
    pos = tau > 0
    return pct_rank(tau[pos]), dlt[pos]


def xy_within_repo(rows: list[dict], variant: str,
                   min_repo: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Percentile computed within (slice, repo): IDF profiles are per-repo,
    so raw tau values are only comparable within a repo. Repos with fewer
    than min_repo tau>0 queries are dropped."""
    by_repo: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if r[variant] > 0:
            qid = r["query_id"]
            repo = qid.rsplit("-pr-", 1)[0] if "-pr-" in qid else qid.rsplit("-", 1)[0]
            by_repo[repo].append((r[variant], r["delta"]))
    xs, ys = [], []
    for pts in by_repo.values():
        if len(pts) < min_repo:
            continue
        t = np.array([p[0] for p in pts])
        d = np.array([p[1] for p in pts])
        xs.append(pct_rank(t))
        ys.append(d)
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def pooled_tests(slices: dict, variant: str, names: list[str],
                 xy_fn=xy) -> dict:
    xs, ys, gi = [], [], []
    for g, name in enumerate(names):
        x, y = xy_fn(slices[name], variant)
        xs.append(x); ys.append(y); gi.append(np.full(len(x), g))
    X, Y, G = map(np.concatenate, (xs, ys, gi))
    k, n = len(names), len(Y)
    D = np.zeros((n, k)); D[np.arange(n), G] = 1.0
    # FE-only vs FE + common slope
    b0, *_ = np.linalg.lstsq(D, Y, rcond=None)
    sse0 = float(((Y - D @ b0) ** 2).sum())
    M1 = np.column_stack([D, X])
    b1, *_ = np.linalg.lstsq(M1, Y, rcond=None)
    sse1 = float(((Y - M1 @ b1) ** 2).sum())
    F = ((sse0 - sse1) / 1) / (sse1 / (n - k - 1))
    p_slope = float(stats.f.sf(F, 1, n - k - 1))
    # slice-cluster bootstrap on common slope: resample slices with replacement
    bs = np.empty(N_BOOT)
    per = [(xs[i], ys[i]) for i in range(k)]
    for b in range(N_BOOT):
        idx = RNG.integers(0, k, k)
        Xb = np.concatenate([per[i][0] for i in idx])
        Yb = np.concatenate([per[i][1] for i in idx])
        Gb = np.concatenate([np.full(len(per[i][0]), j) for j, i in enumerate(idx)])
        Db = np.zeros((len(Yb), k)); Db[np.arange(len(Yb)), Gb] = 1.0
        Mb = np.column_stack([Db, Xb])
        bb, *_ = np.linalg.lstsq(Mb, Yb, rcond=None)
        bs[b] = bb[-1]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"common_slope": float(b1[-1]), "F_slope": float(F),
            "p_slope": p_slope, "cluster_ci": [float(lo), float(hi)],
            "k": k, "n": n}


def r22_r23_forensics(slices: dict, variant: str) -> dict:
    out = {}
    for name in ("py_r23_jina05", "py_r22_jina05", "py_r22_qwen4"):
        rows = slices.get(name)
        if not rows:
            continue
        x, y = xy(rows, variant)
        b, lo, hi = slope_ci(x, y)
        _, qm = bin_means(x, y, 4)
        # per-repo slopes (repo prefix of query_id: '<org>__<repo>-pr-')
        per_repo = defaultdict(list)
        tau = np.array([r[variant] for r in rows])
        for r, t in zip(rows, tau):
            if t > 0:
                per_repo[r["query_id"].rsplit("-pr-", 1)[0]].append(
                    (t, r["delta"]))
        repo_stats = {}
        for repo, pts in sorted(per_repo.items(), key=lambda kv: -len(kv[1])):
            if len(pts) < 8:
                continue
            tt = np.array([p[0] for p in pts]); dd = np.array([p[1] for p in pts])
            repo_stats[repo] = {"n": len(pts),
                                "slope": float(np.polyfit(pct_rank(tt), dd, 1)[0]),
                                "delta_mean": float(dd.mean())}
        out[name] = {"n_tau_pos": len(x), "slope": b, "ci": [lo, hi],
                     "quartile_means": [float(v) for v in qm],
                     "per_repo": repo_stats}
    return out


def collapse_figure(slices: dict, variant: str, names: list[str],
                    xy_fn=xy) -> dict:
    fig, (ax_c, ax_f) = plt.subplots(
        1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})

    # --- panel (a): slice-centered quintile bin means + pooled curve ---
    grid = np.linspace(0.1, 0.9, 5)
    centered_by_slice = []
    for name in names:
        x, y = xy_fn(slices[name], variant)
        bx, by = bin_means(x, y, 5)
        c = by.mean()
        fam = slice_family(name)
        ax_c.plot(bx, by - c, "-", color=FAMILY_COLOR[fam], alpha=0.22, lw=0.8)
        centered_by_slice.append(np.interp(grid, bx, by - c))
    M = np.vstack(centered_by_slice)
    mean_curve = M.mean(axis=0)
    boot = np.empty((N_BOOT, len(grid)))
    for b in range(N_BOOT):
        boot[b] = M[RNG.integers(0, len(M), len(M))].mean(axis=0)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    ax_c.fill_between(grid, lo, hi, color="black", alpha=0.15, zorder=4)
    ax_c.plot(grid, mean_curve, "o-", color="black", lw=2.2, markersize=5,
              zorder=5, label=f"pooled mean over {len(names)} slices (95% CI)")
    ax_c.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax_c.set_xlabel(r"within-slice percentile of $\tau$ (among $\tau>0$)")
    ax_c.set_ylabel(r"$\bar\Delta$ (quintile mean, slice-centered)")
    ax_c.set_title("(a) Premium-decay shape, all slices centered", fontsize=9.5)
    handles = [plt.Line2D([], [], color=FAMILY_COLOR[f], lw=1.6, label=f)
               for f in ("curated", "raw", "raw pre-cutoff", "desc",
                         "ood", "language", "chunk")]
    handles.append(plt.Line2D([], [], color="black", lw=2.2, marker="o",
                              label="pooled mean"))
    ax_c.legend(handles=handles, frameon=False, loc="lower left",
                fontsize=7, ncol=2)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # --- panel (b): slope forest ---
    order = sorted(names, key=lambda n: (slice_family(n), n))
    ys_pos, labels = [], []
    for i, name in enumerate(order):
        x, y = xy_fn(slices[name], variant)
        b, lo_, hi_ = slope_ci(x, y)
        fam = slice_family(name)
        ax_f.errorbar([b], [i], xerr=[[b - lo_], [hi_ - b]], fmt="o",
                      color=FAMILY_COLOR[fam], markersize=3.5, lw=1.1,
                      capsize=2)
        ys_pos.append(i); labels.append(name)
    ax_f.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax_f.set_yticks(ys_pos)
    ax_f.set_yticklabels(labels, fontsize=5.5)
    ax_f.set_xlabel(r"per-slice OLS slope of $\Delta$ on $\tau$-percentile")
    ax_f.set_title("(b) Slope forest (95% bootstrap CI)", fontsize=9.5)
    ax_f.spines["top"].set_visible(False)
    ax_f.spines["right"].set_visible(False)
    ax_f.invert_yaxis()

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plt.savefig(FIG_DIR / f"fig_law_collapse.{ext}", dpi=220,
                    bbox_inches="tight")
    plt.close()
    return {"pooled_curve_x": grid.tolist(),
            "pooled_curve_y": mean_curve.tolist(),
            "pooled_curve_lo": lo.tolist(), "pooled_curve_hi": hi.tolist()}


def main() -> None:
    slices = load_all()
    all_names = sorted(slices)
    core_names = [n for n in all_names if n not in CHUNK_DUPES]
    print(f"loaded {len(all_names)} slices ({len(core_names)} core)\n")

    out: dict = {}
    for variant in VARIANTS:
        block: dict = {}
        for label, names in (("all", all_names), ("core", core_names)):
            t = pooled_tests(slices, variant, names)
            block[f"pooled_{label}"] = t
            print(f"=== {variant} [{label}] ===")
            print(f"  common slope {t['common_slope']:+.4f}  "
                  f"F={t['F_slope']:.1f}  p={t['p_slope']:.2e}  "
                  f"cluster-CI [{t['cluster_ci'][0]:+.4f}, {t['cluster_ci'][1]:+.4f}]")
        t = pooled_tests(slices, variant, core_names, xy_fn=xy_within_repo)
        block["pooled_core_within_repo"] = t
        print(f"  within-repo pct: slope {t['common_slope']:+.4f}  "
              f"p={t['p_slope']:.2e}  "
              f"cluster-CI [{t['cluster_ci'][0]:+.4f}, {t['cluster_ci'][1]:+.4f}]")
        block["forensics"] = r22_r23_forensics(slices, variant)
        out[variant] = block
        print()

    # headline figure: MaxIDF (paper definition) + within-repo percentile
    fig_stats = collapse_figure(slices, "tau_top1", all_names,
                                xy_fn=xy_within_repo)
    out["figure"] = fig_stats
    print("=== r22/r23 forensics (tau_mean5) ===")
    for name, f in out["tau_mean5"]["forensics"].items():
        print(f"  {name}: n={f['n_tau_pos']} slope={f['slope']:+.3f} "
              f"CI [{f['ci'][0]:+.3f},{f['ci'][1]:+.3f}]  "
              f"Q1..Q4 {['%+.3f' % v for v in f['quartile_means']]}")
        for repo, s in list(f["per_repo"].items())[:4]:
            print(f"      {repo:34s} n={s['n']:3d} slope={s['slope']:+.3f} "
                  f"mean={s['delta_mean']:+.3f}")

    (LAW / "law_deepdive.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {LAW / 'law_deepdive.json'}")
    print(f"[fig] wrote {FIG_DIR / 'fig_law_collapse.pdf'}")


if __name__ == "__main__":
    main()
