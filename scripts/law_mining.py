"""Law mining: is the dense-over-BM25 premium governed by one decay curve?

Candidate law: E[Delta | tau] = A_slice * g(tau_rank), where g is a universal
decreasing shape in the within-slice percentile of query distinctiveness tau,
and everything slice-specific (mining regime, encoder, language, repo pool)
enters only through the scalar amplitude A_slice.

Per-slice inputs: per-query smoke files (delta) joined on query_id with
tau_variants files (tau_top1 / tau_jaccard / tau_mean5).

Tests, per tau variant:
  1. per-slice OLS slope of Delta on tau-percentile (tau>0 queries),
     bootstrap CI  -> sign consistency across slices
  2. anchor-shape transfer: fit decile-bin shape g on the anchor slice
     (py_cur_jina05), then per slice fit amplitude-only (multiplicative
     and additive); compare SSE on quintile bin means vs {constant,
     slice-own linear}  -> how much of each slice's tau-structure the
     universal shape explains
  3. slope-homogeneity: pooled regression with slice fixed effects,
     common slope vs per-slice slopes (F-test)

Writes ${REPO_PREMIUM_DATA:-data}/coreb_law/law_fits.json and prints a summary table.

Usage:
  conda run -n llm python scripts/law_mining.py
"""
from __future__ import annotations
from paths import DATA_ROOT

import json
from pathlib import Path

import numpy as np
from scipy import stats

PULL = (DATA_ROOT / 'smoke_test_results')
LAW = (DATA_ROOT / 'analysis/law')
RNG = np.random.default_rng(0)
N_BOOT = 2000
VARIANTS = ("tau_top1", "tau_jaccard", "tau_mean5")
ANCHOR = "py_cur_jina05"


def read_jsonl(f: Path) -> list[dict]:
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def load_slice(result_dir: Path, tau_dir: Path) -> list[dict]:
    """Join per-query delta with tau variants on query_id across repo subdirs."""
    taus: dict[str, dict] = {}
    for f in tau_dir.glob("*/tau_variants.jsonl"):
        for r in read_jsonl(f):
            taus[r["query_id"]] = r
    rows = []
    for f in result_dir.glob("*/smoke_test_django.jsonl"):
        for r in read_jsonl(f):
            tv = taus.get(r["query_id"])
            if tv is None:
                continue
            rows.append({
                "query_id": r["query_id"],
                "delta": r["dense_ndcg10"] - r["bm25_ndcg10"],
                **{v: tv[v] for v in VARIANTS},
            })
    return rows


def slice_table() -> dict[str, tuple[Path, Path]]:
    canon = LAW / "canon"
    tv_ab = {"A": PULL / "tau_variants" / "cellA", "B": PULL / "tau_variants" / "cellB"}
    tvl = LAW / "tau_variants_law"
    s: dict[str, tuple[Path, Path]] = {}
    # jina05 Python cells + special slices
    s["py_cur_jina05"] = (canon / "jina05_cellA_c1000", tv_ab["A"])
    s["py_raw_jina05"] = (canon / "jina05_cellB_c1000", tv_ab["B"])
    s["py_r23_jina05"] = (LAW / "r8_robust_jina-0.5b_c1000", tvl / "r5")
    s["py_r22_jina05"] = (LAW / "r8_2022_jina-0.5b_c1000", tvl / "r8_2022")
    s["py_ood_jina05"] = (LAW / "heldout_jina05_c1000", tvl / "heldout")
    s["py_desc_jina05"] = (LAW / "cellC_jina-0.5b_c1000", tvl / "cellC")
    # chunk sizes (tau joined from c1000 variants; IDF profile approximation)
    for c in (1500, 2000, 3000):
        s[f"py_cur_jina05_c{c}"] = (canon / f"jina05_cellA_c{c}", tv_ab["A"])
    # other languages (jina05)
    s["js_vuejs_jina05"] = (LAW / "r9_vuejs_2023_jina-0.5b_c1000", tvl / "r9_vuejs")
    s["ts_jina05"] = (LAW / "r9b_ts_c1000", tvl / "r9b_ts")
    s["go_jina05"] = (LAW / "r9c_go_c1000", tvl / "r9c_go")
    s["rust_jina05"] = (LAW / "r9d_rust_c1000", tvl / "r9d_rust")
    # other encoders x Python cells
    for tag in ("jina15", "qwen06", "qwen4", "qwen8", "c2llm05", "c2llm7",
                "f2llm06", "f2llm4", "e5base", "e5large", "gteq15", "gteq7"):
        s[f"py_cur_{tag}"] = (canon / f"{tag}_cellA_c1000", tv_ab["A"])
        s[f"py_raw_{tag}"] = (canon / f"{tag}_cellB_c1000", tv_ab["B"])
    # extra encoder on r22, and Desc cell encoders
    s["py_r22_qwen4"] = (LAW / "r8_2022_qwen3-4b_c1000", tvl / "r8_2022")
    s["py_desc_jina15"] = (LAW / "cellC_jina-1.5b_c1000", tvl / "cellC")
    s["py_desc_qwen06"] = (LAW / "cellC_Qwen3-0.6B_c1000", tvl / "cellC")
    s["py_desc_qwen4"] = (LAW / "cellC_Qwen3-4B_c1000", tvl / "cellC")
    return s


def pct_rank(x: np.ndarray) -> np.ndarray:
    """Within-slice percentile rank in (0, 1]."""
    return stats.rankdata(x) / len(x)


def slope_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    b = np.polyfit(x, y, 1)[0]
    bs = np.empty(N_BOOT)
    n = len(x)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        bs[i] = np.polyfit(x[idx], y[idx], 1)[0]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(b), float(lo), float(hi)


def bin_means(x: np.ndarray, y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(x, np.linspace(0, 1, k + 1))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, k - 1)
    cx = np.array([x[idx == i].mean() for i in range(k) if (idx == i).any()])
    cy = np.array([y[idx == i].mean() for i in range(k) if (idx == i).any()])
    return cx, cy


def main() -> None:
    slices = {}
    for name, (rdir, tdir) in slice_table().items():
        if not rdir.exists() or not tdir.exists():
            print(f"[skip] {name}: missing {rdir if not rdir.exists() else tdir}")
            continue
        rows = load_slice(rdir, tdir)
        if len(rows) < 20:
            print(f"[skip] {name}: only {len(rows)} joined rows")
            continue
        slices[name] = rows
    print(f"loaded {len(slices)} slices, "
          f"{sum(len(r) for r in slices.values())} query-slice rows\n")

    out: dict = {"variants": {}}
    for variant in VARIANTS:
        per_slice = {}
        # ---- per-slice slopes on tau>0 percentile ----
        for name, rows in slices.items():
            tau = np.array([r[variant] for r in rows])
            dlt = np.array([r["delta"] for r in rows])
            pos = tau > 0
            n_pos = int(pos.sum())
            entry = {"n": len(rows), "n_tau_pos": n_pos,
                     "delta_mean": float(dlt.mean()),
                     "tau0_frac": float(1 - pos.mean()),
                     "delta_tau0": float(dlt[~pos].mean()) if (~pos).any() else None}
            if n_pos >= 20:
                x = pct_rank(tau[pos])
                y = dlt[pos]
                b, lo, hi = slope_ci(x, y)
                entry.update(slope=b, slope_lo=lo, slope_hi=hi,
                             pearson_r=float(stats.pearsonr(x, y)[0]))
            per_slice[name] = entry

        # ---- anchor shape (decile bin means on percentile) ----
        anc = slices[ANCHOR]
        tau_a = np.array([r[variant] for r in anc])
        dlt_a = np.array([r["delta"] for r in anc])
        pos_a = tau_a > 0
        gx, gy = bin_means(pct_rank(tau_a[pos_a]), dlt_a[pos_a], 10)

        def g(xq: np.ndarray) -> np.ndarray:
            return np.interp(xq, gx, gy)

        # ---- amplitude-only transfer, scored on quintile bin means ----
        transfer = {}
        for name, rows in slices.items():
            tau = np.array([r[variant] for r in rows])
            dlt = np.array([r["delta"] for r in rows])
            pos = tau > 0
            if pos.sum() < 25:
                continue
            x, y = pct_rank(tau[pos]), dlt[pos]
            bx, by = bin_means(x, y, 5)
            gb = g(bx)
            # multiplicative amplitude
            A = float(np.dot(gb, by) / max(np.dot(gb, gb), 1e-12))
            sse_mult = float(((by - A * gb) ** 2).sum())
            # additive offset
            c = float((by - gb).mean())
            sse_add = float(((by - (gb + c)) ** 2).sum())
            # baselines
            sse_const = float(((by - by.mean()) ** 2).sum())
            coef = np.polyfit(bx, by, 1)
            sse_lin = float(((by - np.polyval(coef, bx)) ** 2).sum())
            transfer[name] = {
                "A_mult": A, "c_add": c,
                "sse_mult": sse_mult, "sse_add": sse_add,
                "sse_const": sse_const, "sse_own_linear": sse_lin,
                "r2_shape_mult": 1 - sse_mult / max(sse_const, 1e-12),
                "r2_shape_add": 1 - sse_add / max(sse_const, 1e-12),
            }

        # ---- slope homogeneity (pooled, slice fixed effects) ----
        names = [n for n, e in per_slice.items() if "slope" in e]
        xs, ys, gs_idx = [], [], []
        for gi, name in enumerate(names):
            rows = slices[name]
            tau = np.array([r[variant] for r in rows])
            dlt = np.array([r["delta"] for r in rows])
            pos = tau > 0
            xs.append(pct_rank(tau[pos]))
            ys.append(dlt[pos])
            gs_idx.append(np.full(int(pos.sum()), gi))
        X = np.concatenate(xs)
        Y = np.concatenate(ys)
        G = np.concatenate(gs_idx)
        k = len(names)
        n_tot = len(Y)
        # common-slope model: Y ~ FE + b*X
        D = np.zeros((n_tot, k))
        D[np.arange(n_tot), G] = 1.0
        M1 = np.column_stack([D, X])
        b1, res1, *_ = np.linalg.lstsq(M1, Y, rcond=None)
        sse1 = float(((Y - M1 @ b1) ** 2).sum())
        # per-slice-slope model: Y ~ FE + b_g*X
        M2 = np.column_stack([D, D * X[:, None]])
        b2, *_ = np.linalg.lstsq(M2, Y, rcond=None)
        sse2 = float(((Y - M2 @ b2) ** 2).sum())
        df1, df2 = k - 1, n_tot - 2 * k
        F = ((sse1 - sse2) / df1) / (sse2 / df2)
        p_homog = float(stats.f.sf(F, df1, df2))
        common_slope = float(b1[-1])

        out["variants"][variant] = {
            "per_slice": per_slice, "transfer": transfer,
            "anchor_shape_x": gx.tolist(), "anchor_shape_y": gy.tolist(),
            "homogeneity": {"common_slope": common_slope, "F": float(F),
                            "p": p_homog, "k_slices": k, "n": n_tot},
        }

        # ---- print summary ----
        neg = sum(1 for n in names if per_slice[n]["slope"] < 0)
        ci_neg = sum(1 for n in names if per_slice[n]["slope_hi"] < 0)
        med_r2_add = float(np.median([t["r2_shape_add"] for t in transfer.values()]))
        med_r2_mult = float(np.median([t["r2_shape_mult"] for t in transfer.values()]))
        print(f"=== {variant} ===")
        print(f"  slices with slope<0: {neg}/{len(names)}  (CI excludes 0: {ci_neg})")
        print(f"  common slope: {common_slope:+.4f}   "
              f"homogeneity F={F:.2f} p={p_homog:.4f} (p>0.05 => shapes compatible)")
        print(f"  median R2 of anchor shape on other slices: "
              f"add {med_r2_add:+.3f}, mult {med_r2_mult:+.3f}")
        worst = sorted(names, key=lambda n: per_slice[n]["slope"])[-3:]
        print(f"  least-negative slopes: "
              + ", ".join(f"{n} {per_slice[n]['slope']:+.3f}" for n in worst))
        print()

    (LAW / "law_fits.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {LAW / 'law_fits.json'}")


if __name__ == "__main__":
    main()
