#!/usr/bin/env python3
"""Aggregate R7 graph-as-rerank outputs into the numbers the paper reports.

Reads the per-query files written by graph_rerank.py, laid out as
    <root>/cell<CELL>/<encoder>/<repo>.jsonl
and reports, per cell x encoder: mean dense nDCG@10, mean graph-rerank
nDCG@10, their paired difference with a repository-clustered bootstrap CI,
and the per-query better/worse counts.

Repository-clustered rather than per-query: queries from one repository share
a corpus and a graph, so they are not independent draws.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load(root: Path) -> dict[tuple[str, str], dict[str, list[tuple[float, float]]]]:
    """-> {(cell, encoder): {repo: [(dense, rerank), ...]}}"""
    out: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(root.glob("cell*/*/*.jsonl")):
        cell = f.parent.parent.name.replace("cell", "")
        enc, repo = f.parent.name, f.stem
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            d, g = r.get("dense_ndcg10"), r.get("graph_rerank_ndcg10")
            if d is not None and g is not None:
                out[(cell, enc)][repo].append((d, g))
    return out


def boot(by_repo: dict[str, list[tuple[float, float]]], n_boot: int, seed: int) -> tuple[float, float, float]:
    repos = sorted(by_repo)
    flat = [p for r in repos for p in by_repo[r]]
    point = sum(g - d for d, g in flat) / len(flat)
    rnd = random.Random(seed)
    means = []
    for _ in range(n_boot):
        pairs = [p for _ in repos for p in by_repo[repos[rnd.randrange(len(repos))]]]
        means.append(sum(g - d for d, g in pairs) / len(pairs))
    means.sort()
    return point, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()

    data = load(args.root)
    if not data:
        print(f"no per-query files under {args.root}")
        return 1
    report = {}
    print(f"{'cell':5s} {'encoder':16s} {'n':>5s} {'dense':>7s} {'rerank':>7s} {'delta':>8s} "
          f"{'95% CI':>20s}  better/worse")
    for (cell, enc), by_repo in sorted(data.items()):
        flat = [p for r in by_repo for p in by_repo[r]]
        dense = sum(d for d, _ in flat) / len(flat)
        rerank = sum(g for _, g in flat) / len(flat)
        point, lo, hi = boot(by_repo, args.n_boot, args.seed)
        better = sum(1 for d, g in flat if g > d)
        worse = sum(1 for d, g in flat if g < d)
        print(f"{cell:5s} {enc:16s} {len(flat):5d} {dense:7.4f} {rerank:7.4f} {point:+8.4f} "
              f"[{lo:+7.4f},{hi:+7.4f}]  {better}/{worse}")
        report[f"cell{cell}|{enc}"] = {
            "n_queries": len(flat), "n_repos": len(by_repo),
            "dense_ndcg10": round(dense, 4), "graph_rerank_ndcg10": round(rerank, 4),
            "delta": round(point, 4), "ci": [round(lo, 4), round(hi, 4)],
            "better": better, "worse": worse,
        }
    # pooled per cell, across encoders
    for cell in sorted({c for c, _ in data}):
        merged: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for (c, _enc), by_repo in data.items():
            if c != cell:
                continue
            for repo, pairs in by_repo.items():
                merged[repo].extend(pairs)
        point, lo, hi = boot(merged, args.n_boot, args.seed)
        print(f"POOLED cell{cell}: delta {point:+.4f} [{lo:+.4f},{hi:+.4f}]")
        report[f"cell{cell}|POOLED"] = {"delta": round(point, 4), "ci": [round(lo, 4), round(hi, 4)]}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
