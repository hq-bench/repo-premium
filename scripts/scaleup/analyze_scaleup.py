#!/usr/bin/env python3
"""Confirmatory analysis of the matched scale-up.

The original construction estimated the mining-methodology effect by comparing
two different *pools* of queries (curated SWE-bench Verified problems vs raw
post-cutoff PRs) drawn from different tasks in ten Python repositories. That
comparison is between-task, so query mining is entangled with whatever else
differs between the pools.

This design removes that entanglement. Every task carries two query views of
the *same* pull request against the *same* gold files and the *same* corpus
snapshot, so the mining effect is estimated strictly within task. Date is
varied separately and blocked by repository, giving the full matched 2x2:

                pre-cutoff code   post-cutoff code
    curated       Cur/Pre            Cur/Post
    raw PR text   Raw/Pre            Raw/Post

Q2 predicts a large mining effect and a null date effect. Both are tested here.

Confidence intervals use a hierarchical bootstrap: resample repositories with
replacement, then tasks within each drawn repository, matching the nesting of
the design. The paired mining contrast is differenced within task before
resampling, so it is not contaminated by between-task variance.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def hierarchical_bootstrap(
    by_repo: dict[str, list[float]], n_boot: int, seed: int,
    stat: Callable[[list[float]], float] = mean,
) -> tuple[float, float, float]:
    """Resample repositories, then tasks within each drawn repository."""
    repos = list(by_repo)
    flat = [v for vs in by_repo.values() for v in vs]
    if not flat:
        return float("nan"), float("nan"), float("nan")
    point = stat(flat)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in range(len(repos)):
            values = by_repo[repos[rng.randrange(len(repos))]]
            sample.extend(values[rng.randrange(len(values))] for _ in range(len(values)))
        if sample:
            draws.append(stat(sample))
    draws.sort()
    return (point, draws[int(0.025 * len(draws))],
            draws[min(len(draws) - 1, int(0.975 * len(draws)))])


def group(rows: list[dict[str, Any]], key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            out[row["repo"]].append(row[key])
    return dict(out)


def summarise(rows: list[dict[str, Any]], key: str, n_boot: int, seed: int) -> dict[str, Any]:
    point, lo, hi = hierarchical_bootstrap(group(rows, key), n_boot, seed)
    return {"mean": round(point, 4), "ci": [round(lo, 4), round(hi, 4)], "n": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    meta = {row["task_id"]: row for row in read_jsonl(args.manifest)}
    tasks: dict[str, dict[str, Any]] = {}
    for task_dir in sorted(p for p in args.retrieval_root.iterdir() if p.is_dir()):
        record_path = task_dir / "smoke_test_django.jsonl"
        if not record_path.is_file():
            continue
        info = meta.get(task_dir.name)
        if info is None:
            continue
        views: dict[str, dict[str, Any]] = {}
        for record in read_jsonl(record_path):
            view = record["query_id"].rsplit("::", 1)[-1]
            views[view] = record
        if {"curated", "raw"} - set(views):
            continue
        tasks[task_dir.name] = {"info": info, "views": views}

    rows: list[dict[str, Any]] = []
    for task_id, payload in tasks.items():
        info, views = payload["info"], payload["views"]
        cur, raw = views["curated"], views["raw"]
        d_cur = cur["dense_ndcg10"] - cur["bm25_ndcg10"]
        d_raw = raw["dense_ndcg10"] - raw["bm25_ndcg10"]
        rows.append({
            "task_id": task_id,
            "repo": info["repo"],
            "language": info["language_group"],
            "window": info["date_window"],
            "split": info["split"],
            "delta_curated": d_cur,
            "delta_raw": d_raw,
            # The paired mining contrast: same task, same gold, same corpus.
            "mining_effect": d_raw - d_cur,
            "bm25_curated": cur["bm25_ndcg10"],
            "bm25_raw": raw["bm25_ndcg10"],
            "dense_curated": cur["dense_ndcg10"],
            "dense_raw": raw["dense_ndcg10"],
            "tau_curated": cur.get("tau_q"),
            "tau_raw": raw.get("tau_q"),
        })

    report: dict[str, Any] = {
        "n_tasks": len(rows),
        "n_repositories": len({r["repo"] for r in rows}),
        "n_query_views": 2 * len(rows),
        "n_boot": args.n_boot,
        "bootstrap": "hierarchical (repository, then task within repository)",
    }

    # --- Q2: mining effect, estimated within task -------------------------
    report["Q2_mining_effect_within_task"] = summarise(rows, "mining_effect", args.n_boot, args.seed)
    report["premium_by_view"] = {
        "curated": summarise(rows, "delta_curated", args.n_boot, args.seed + 1),
        "raw": summarise(rows, "delta_raw", args.n_boot, args.seed + 2),
    }
    report["mean_ndcg10"] = {
        k: round(mean([r[k] for r in rows]), 4)
        for k in ("bm25_curated", "bm25_raw", "dense_curated", "dense_raw")
    }

    # --- Date effect, blocked by repository -------------------------------
    date_effects: dict[str, Any] = {}
    for view in ("curated", "raw"):
        key = f"delta_{view}"
        pre = [r for r in rows if r["window"] == "pre"]
        post = [r for r in rows if r["window"] == "post"]
        p_pre = summarise(pre, key, args.n_boot, args.seed + 3)
        p_post = summarise(post, key, args.n_boot, args.seed + 4)
        # Difference within repository, so date is not confounded by repo mix.
        by_repo_diff: dict[str, list[float]] = {}
        for repo in {r["repo"] for r in rows}:
            a = [r[key] for r in pre if r["repo"] == repo]
            b = [r[key] for r in post if r["repo"] == repo]
            if a and b:
                by_repo_diff[repo] = [mean(b) - mean(a)]
        pt, lo, hi = hierarchical_bootstrap(by_repo_diff, args.n_boot, args.seed + 5)
        date_effects[view] = {
            "pre": p_pre, "post": p_post,
            "post_minus_pre_within_repo": {
                "mean": round(pt, 4), "ci": [round(lo, 4), round(hi, 4)],
                "n_repositories": len(by_repo_diff),
            },
        }
    report["date_effect"] = date_effects

    # --- Slices ------------------------------------------------------------
    for name, keyfn in (("by_language", lambda r: r["language"]),
                        ("by_split", lambda r: r["split"]),
                        ("by_window", lambda r: r["window"])):
        out: dict[str, Any] = {}
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[keyfn(row)].append(row)
        for label, subset in sorted(buckets.items()):
            out[label] = {
                "n_tasks": len(subset),
                "n_repositories": len({r["repo"] for r in subset}),
                "mining_effect": summarise(subset, "mining_effect", args.n_boot, args.seed + 6),
                "delta_curated": round(mean([r["delta_curated"] for r in subset]), 4),
                "delta_raw": round(mean([r["delta_raw"] for r in subset]), 4),
            }
        report[name] = out

    # Sign consistency across repositories: how many repositories show the
    # mining effect in the predicted direction at all.
    per_repo = defaultdict(list)
    for row in rows:
        per_repo[row["repo"]].append(row["mining_effect"])
    positives = sum(1 for vs in per_repo.values() if mean(vs) > 0)
    report["repositories_with_positive_mining_effect"] = {
        "n_positive": positives, "n_repositories": len(per_repo),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
