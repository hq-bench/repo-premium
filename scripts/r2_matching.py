"""R2 — length/scope matching between Cur-Pre (cell A) and Raw-Post (cell B).

The curated-vs-raw premium gap (circle-2) could in principle be a
length/scope artifact: raw PR bodies are shorter and touch more files.
R2 re-estimates the gap after matching the two cells on
(z-scored stripped-query length, n_gold_files) with three strategies:

  1. nn      — for each B query, the nearest A query in matching space
               (with replacement); gap = mean(delta_B) - mean(delta_A|matched)
  2. clip    — restrict both cells to the overlapping support
               (5th-95th percentile box of the other cell)
  3. strata  — reweight A to B's cell distribution over
               (length tercile x n_gold bucket) strata

Reads ${REPO_PREMIUM_DATA:-data}/coreb_pull/{cellA,cellB} per-query results and
${REPO_PREMIUM_DATA:-data}/coreb_pull/{qrelsA,qrelsB} qrel files (for query text / n_gold).

Usage:
  conda run -n llm python scripts/r2_matching.py
"""
from __future__ import annotations
from paths import DATA_ROOT

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from run_retrieval import strip_pr_boilerplate  # noqa: E402

PULL = (DATA_ROOT / 'smoke_test_results')
RNG = np.random.default_rng(0)
N_BOOT = 10_000


def load(cell: str, qdir: str) -> list[dict]:
    meta = {}
    for f in (PULL / qdir).glob("*.jsonl"):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                meta[r["query_id"]] = {
                    "len": len(strip_pr_boilerplate(r.get("problem_statement", ""))),
                    "n_gold": r.get("n_gold_files", len(r.get("gold_files", []))),
                }
    rows = []
    for f in (PULL / cell).glob("*/smoke_test_django.jsonl"):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                m = meta.get(r["query_id"])
                if m is None:
                    continue
                rows.append({"delta": r["dense_ndcg10"] - r["bm25_ndcg10"], **m})
    return rows


def boot_gap(db: np.ndarray, da: np.ndarray) -> tuple[float, float, float]:
    g = db.mean() - da.mean()
    bs = [db[RNG.integers(0, len(db), len(db))].mean()
          - da[RNG.integers(0, len(da), len(da))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(g), float(lo), float(hi)


def main() -> None:
    A = load("cellA", "qrelsA")
    B = load("cellB", "qrelsB")
    lenA = np.array([r["len"] for r in A]); lenB = np.array([r["len"] for r in B])
    ngA = np.array([r["n_gold"] for r in A]); ngB = np.array([r["n_gold"] for r in B])
    dA = np.array([r["delta"] for r in A]); dB = np.array([r["delta"] for r in B])
    print(f"A n={len(A)} mean len={lenA.mean():.0f} multi-file={100*(ngA>1).mean():.0f}%")
    print(f"B n={len(B)} mean len={lenB.mean():.0f} multi-file={100*(ngB>1).mean():.0f}%")

    g, lo, hi = boot_gap(dB, dA)
    print(f"unmatched gap: {g:+.3f}  CI [{lo:+.3f},{hi:+.3f}]")

    # z-space over the pooled sample
    all_len = np.concatenate([lenA, lenB])
    mu, sd = all_len.mean(), all_len.std()
    zA, zB = (lenA - mu) / sd, (lenB - mu) / sd

    # 1) NN matching (features: z-length, n_gold scaled by pooled std)
    ng_sd = np.concatenate([ngA, ngB]).std()
    FA = np.column_stack([zA, ngA / ng_sd])
    FB = np.column_stack([zB, ngB / ng_sd])
    idx = np.array([np.argmin(((FA - fb) ** 2).sum(1)) for fb in FB])
    g, lo, hi = boot_gap(dB, dA[idx])
    print(f"NN-matched gap: {g:+.3f}  CI [{lo:+.3f},{hi:+.3f}]  "
          f"(unique A matches: {len(set(idx))})")

    # 2) range clip to mutual 5-95 pct box
    def box(x, lo_p=5, hi_p=95):
        return np.percentile(x, lo_p), np.percentile(x, hi_p)
    l_lo, l_hi = max(box(lenA)[0], box(lenB)[0]), min(box(lenA)[1], box(lenB)[1])
    g_lo, g_hi = max(ngA.min(), ngB.min()), min(box(ngA)[1], box(ngB)[1])
    mA = (lenA >= l_lo) & (lenA <= l_hi) & (ngA >= g_lo) & (ngA <= g_hi)
    mB = (lenB >= l_lo) & (lenB <= l_hi) & (ngB >= g_lo) & (ngB <= g_hi)
    g, lo, hi = boot_gap(dB[mB], dA[mA])
    print(f"range-clip gap: {g:+.3f}  CI [{lo:+.3f},{hi:+.3f}]  "
          f"(A n={int(mA.sum())}, B n={int(mB.sum())})")

    # 3) stratified reweighting of A to B's strata
    ters = np.percentile(np.concatenate([lenA, lenB]), [33.3, 66.7])
    def stratum(ln, ng):
        t = int(np.digitize(ln, ters))
        gbin = 0 if ng == 1 else (1 if ng <= 3 else 2)
        return t * 3 + gbin
    sA = np.array([stratum(l, n) for l, n in zip(lenA, ngA)])
    sB = np.array([stratum(l, n) for l, n in zip(lenB, ngB)])
    means, weights = [], []
    for s in sorted(set(sB)):
        wa = (sA == s)
        if not wa.any():
            continue
        means.append(dA[wa].mean())
        weights.append((sB == s).mean())
    w = np.array(weights) / np.sum(weights)
    a_rw = float(np.dot(w, means))
    g = float(dB.mean() - a_rw)
    print(f"stratified-resample gap: {g:+.3f}  "
          f"(A reweighted mean {a_rw:+.3f}; strata covered "
          f"{np.sum(weights)*100:.0f}% of B mass)")


if __name__ == "__main__":
    main()
