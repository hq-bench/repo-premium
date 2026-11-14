#!/usr/bin/env python3
"""Per-query mechanism analysis for the generative arm.

The paper claims the anchors close the premium because BM25 can *match* them,
not because they move the query along the tau curve. This script converts that
claim from cell-level inference into a per-query measurement.

For each query in each cell it computes, on the cached genarm data:

  bm25_gain    = bm25_ndcg10(expanded) - bm25_ndcg10(control)      [outcome]
  tau_shift    = tau_q(expanded) - tau_q(control)                  [the max-based predictor the paper rejects]
  added tokens = tokenize(generated_query) - tokenize(stripped statement)
                 (the NEW lexical evidence generation contributed)
  idf_sum_hit  = sum of chunk-level IDF over added tokens present in the gold
                 files                                             [the sum-based predictor the mechanism implies]
  count_hit, frac_hit = |added ∩ gold|, and that as a fraction of |added|

The design point: tau is a MAX over shared tokens, BM25 scores a SUM over
matching terms. If the mechanism is "generation supplies matchable evidence",
idf_sum_hit should predict the per-query BM25 gain where tau_shift does not.

Fidelity: tokenize/list_corpus/read_file/chunk_file/compute_idf are imported
from the retrieval script itself (the implementation the generative arms used,
chunk_chars=1000), so the IDF table is the one the cached runs used, by
construction. Pure CPU; no GPU, no re-retrieval.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

CELLS = ("A", "B", "RA", "RB")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def boot_ci_r(xs: list[float], ys: list[float], n_boot: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(xs)
    draws = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        r = pearson([xs[i] for i in idx], [ys[i] for i in idx])
        if not math.isnan(r):
            draws.append(r)
    draws.sort()
    if len(draws) < 100:
        return [float("nan"), float("nan")]
    return [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]]


def ols2_standardized(y: list[float], x1: list[float], x2: list[float]) -> dict[str, float]:
    """Standardized coefficients for y ~ x1 + x2 (closed form, two regressors)."""
    def z(v: list[float]) -> list[float]:
        m = sum(v) / len(v)
        s = math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1)) or 1.0
        return [(a - m) / s for a in v]
    zy, z1, z2 = z(y), z(x1), z(x2)
    r12 = pearson(z1, z2)
    ry1 = pearson(zy, z1)
    ry2 = pearson(zy, z2)
    denom = 1 - r12 ** 2
    if abs(denom) < 1e-9:
        return {"beta_x1": float("nan"), "beta_x2": float("nan"), "r_x1_x2": r12}
    return {
        "beta_x1": (ry1 - r12 * ry2) / denom,
        "beta_x2": (ry2 - r12 * ry1) / denom,
        "r_x1_x2": r12,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genarm-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True,
                        help="repo checkout containing scripts/smoke_test_django.py")
    parser.add_argument("--chunk-chars", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    # Import the retrieval script's own primitives for exact fidelity.
    sys.path.insert(0, str(args.code_root / "scripts"))
    from run_retrieval import (  # type: ignore
        tokenize, list_corpus, read_file, chunk_file, compute_idf,
    )

    # jobs.tsv: arm \t cell \t repo \t qrel_path \t clone_dir
    clone_dirs: dict[tuple[str, str], Path] = {}
    for line in (args.genarm_root / "jobs.tsv").read_text().splitlines():
        arm, cell, repo, _qrel, clone = line.split("\t")
        clone_dirs[(cell, repo)] = Path(clone)

    def load_results(arm: str, cell: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        root = args.genarm_root / "results" / f"{arm}_cell{cell}"
        for repo_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for rec in read_jsonl(repo_dir / "smoke_test_django.jsonl"):
                rec["_repo"] = repo_dir.name
                out[rec["query_id"]] = rec
        return out

    report: dict[str, Any] = {"chunk_chars": args.chunk_chars, "by_cell": {}}
    # Keyed by clone path: cellA and cellB share snapshots, so this halves the builds.
    idf_cache: dict[str, dict[str, float]] = {}

    for cell in CELLS:
        control = load_results("control", cell)
        expanded = load_results("expanded", cell)
        qrel_dir = args.genarm_root / f"qrels_cell{cell}_expanded"

        rows: list[dict[str, Any]] = []
        n_fallback = 0
        for qrel_path in sorted(qrel_dir.glob("*.jsonl")):
            repo = qrel_path.stem
            clone = clone_dirs.get((cell, repo))
            if clone is None or not clone.is_dir():
                print(f"[warn] no clone for cell{cell} {repo}", file=sys.stderr)
                continue
            key = str(clone)
            if key not in idf_cache:
                print(f"[idf] cell{cell} {repo} ...", file=sys.stderr, flush=True)
                paths = list_corpus(clone)
                chunks: list[str] = []
                for p in paths:
                    chunks.extend(chunk_file(read_file(clone, p), args.chunk_chars))
                idf_cache[key] = compute_idf([tokenize(c) for c in chunks])
            idf = idf_cache[key]

            for row in read_jsonl(qrel_path):
                qid = row["query_id"]
                c, e = control.get(qid), expanded.get(qid)
                if c is None or e is None:
                    continue
                gen_q = (row.get("generated_query") or "").strip()
                orig = row.get("problem_statement_stripped") or row.get("problem_statement") or ""
                added = set(tokenize(gen_q)) - set(tokenize(orig))
                if not gen_q or not added:
                    n_fallback += 1
                    continue
                gold_tokens: set[str] = set()
                for g in row["gold_files"]:
                    gold_tokens |= set(tokenize(read_file(clone, g)))
                hit = added & gold_tokens
                rows.append({
                    "query_id": qid,
                    "repo": repo,
                    "bm25_gain": e["bm25_ndcg10"] - c["bm25_ndcg10"],
                    "tau_shift": e["tau_q"] - c["tau_q"],
                    "n_added": len(added),
                    "count_hit": len(hit),
                    "frac_hit": len(hit) / len(added),
                    "idf_sum_hit": sum(idf.get(t, 0.0) for t in hit),
                })

        if not rows:
            report["by_cell"][cell] = {"error": "no rows"}
            continue
        gain = [r["bm25_gain"] for r in rows]
        entry: dict[str, Any] = {"n": len(rows), "n_fallback_excluded": n_fallback}
        for pred in ("idf_sum_hit", "count_hit", "frac_hit", "tau_shift"):
            xs = [r[pred] for r in rows]
            entry[pred] = {
                "pearson_r": round(pearson(xs, gain), 3),
                "pearson_ci": [round(v, 3) for v in boot_ci_r(xs, gain, args.n_boot, args.seed)],
                "spearman_rho": round(spearman(xs, gain), 3),
            }
        entry["ols_gain_on_idfsum_and_taushift"] = {
            k: round(v, 3) for k, v in ols2_standardized(
                gain, [r["idf_sum_hit"] for r in rows], [r["tau_shift"] for r in rows]
            ).items()
        }
        # Tercile contrast on the headline predictor, for a table-friendly number.
        srt = sorted(rows, key=lambda r: r["idf_sum_hit"])
        third = len(srt) // 3
        lo, hi = srt[:third], srt[-third:]
        entry["gain_bottom_tercile_idfsum"] = round(sum(r["bm25_gain"] for r in lo) / len(lo), 4)
        entry["gain_top_tercile_idfsum"] = round(sum(r["bm25_gain"] for r in hi) / len(hi), 4)
        report["by_cell"][cell] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
