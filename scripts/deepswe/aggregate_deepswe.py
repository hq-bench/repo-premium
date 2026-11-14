#!/usr/bin/env python3
"""Aggregate DeepSWE retrieval into the D1-D3 external-validation answers.

D1  External validity: does the dense-over-BM25 premium persist on DeepSWE's
    original, long-horizon, multilingual tasks?
D2  Lexical rescue: how often does BM25 beat dense retrieval, and where?
D3  Query-distinctiveness dependence: does the premium still decay with tau?

DeepSWE contributes 113 tasks across 91 repositories, so most repositories
supply a single task. The main paper's *within-repository* tau-percentile law
is therefore not reproducible here and is not attempted; tau is pooled and
reported as such.

Confidence intervals are repository-clustered paired bootstrap: repositories,
not queries, are the resampling unit, because tasks inside a repository share a
corpus and an IDF profile.
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


def cluster_bootstrap(
    clusters: list[list[float]], stat: Callable[[list[float]], float],
    n_boot: int, seed: int,
) -> tuple[float, float, float]:
    """Percentile CI resampling whole clusters with replacement."""
    flat = [v for c in clusters for v in c]
    if not flat:
        return float("nan"), float("nan"), float("nan")
    point = stat(flat)
    rng = random.Random(seed)
    draws = []
    n = len(clusters)
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in range(n):
            sample.extend(clusters[rng.randrange(n)])
        if sample:
            draws.append(stat(sample))
    draws.sort()
    if not draws:
        return point, float("nan"), float("nan")
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return point, lo, hi


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def load_encoder(root: Path, manifest: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        record_path = task_dir / "smoke_test_django.jsonl"
        if not record_path.is_file():
            continue
        summary_path = task_dir / "smoke_test_django_summary.json"
        n_corpus = None
        if summary_path.is_file():
            n_corpus = json.loads(summary_path.read_text()).get("n_corpus_files")
        for record in read_jsonl(record_path):
            meta = manifest.get(record["query_id"])
            if meta is None:
                continue
            rows.append({
                "task_id": record["query_id"],
                "repo": meta["repo"],
                "language": meta["language"],
                "tau": record.get("tau_q"),
                "dense": record.get("dense_ndcg10", 0.0),
                "bm25": record.get("bm25_ndcg10", 0.0),
                "grep": record.get("grep_ndcg10", 0.0),
                "dense_recall": record.get("dense_recall10"),
                "bm25_recall": record.get("bm25_recall10"),
                "n_gold": len(record.get("gold_files", [])),
                "n_corpus_files": n_corpus,
            })
    for row in rows:
        row["delta"] = row["dense"] - row["bm25"]
        row["delta_grep"] = row["dense"] - row["grep"]
        row["oracle"] = max(row["dense"], row["bm25"])
    return rows


def by_repo(rows: list[dict[str, Any]], key: str) -> list[list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[row["repo"]].append(row[key])
    return list(groups.values())


def quartile_report(rows: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any]:
    scored = [r for r in rows if r["tau"] is not None and r["tau"] > 0]
    zero = [r for r in rows if r["tau"] is not None and r["tau"] <= 0]
    scored.sort(key=lambda r: r["tau"])
    out: dict[str, Any] = {
        "n_tau_positive": len(scored),
        "n_tau_zero": len(zero),
        "tau_zero_mean_delta": round(mean([r["delta"] for r in zero]), 4) if zero else None,
        "quartiles": [],
    }
    if len(scored) < 8:
        return out
    size = len(scored) // 4
    for q in range(4):
        lo = q * size
        hi = (q + 1) * size if q < 3 else len(scored)
        chunk = scored[lo:hi]
        point, cl, ch = cluster_bootstrap(by_repo(chunk, "delta"), mean, n_boot, seed + q)
        out["quartiles"].append({
            "quartile": q + 1,
            "n": len(chunk),
            "tau_range": [round(chunk[0]["tau"], 3), round(chunk[-1]["tau"], 3)],
            "mean_delta": round(point, 4),
            "ci": [round(cl, 4), round(ch, 4)],
            "bm25_win_rate": round(mean([1.0 if r["bm25"] > r["dense"] else 0.0 for r in chunk]), 4),
        })
    first, last = out["quartiles"][0]["mean_delta"], out["quartiles"][-1]["mean_delta"]
    out["q1_minus_q4"] = round(first - last, 4)
    out["monotone_decreasing"] = all(
        out["quartiles"][i]["mean_delta"] >= out["quartiles"][i + 1]["mean_delta"]
        for i in range(3)
    )
    return out


def encoder_report(
    name: str, rows: list[dict[str, Any]], n_boot: int, seed: int
) -> dict[str, Any]:
    delta, dlo, dhi = cluster_bootstrap(by_repo(rows, "delta"), mean, n_boot, seed)
    dg, glo, ghi = cluster_bootstrap(by_repo(rows, "delta_grep"), mean, n_boot, seed + 1)
    bm25_wins = [r for r in rows if r["bm25"] > r["dense"]]
    ties = [r for r in rows if r["bm25"] == r["dense"]]
    report: dict[str, Any] = {
        "encoder": name,
        "n_queries": len(rows),
        "n_repositories": len({r["repo"] for r in rows}),
        "mean_ndcg10": {
            "dense": round(mean([r["dense"] for r in rows]), 4),
            "bm25": round(mean([r["bm25"] for r in rows]), 4),
            "grep": round(mean([r["grep"] for r in rows]), 4),
            "oracle": round(mean([r["oracle"] for r in rows]), 4),
        },
        "D1_premium_dense_minus_bm25": {
            "mean": round(delta, 4), "ci": [round(dlo, 4), round(dhi, 4)],
            "positive": dlo > 0,
        },
        "premium_dense_minus_grep": {
            "mean": round(dg, 4), "ci": [round(glo, 4), round(ghi, 4)],
        },
        "D2_lexical_rescue": {
            "bm25_win_rate": round(len(bm25_wins) / len(rows), 4) if rows else None,
            "tie_rate": round(len(ties) / len(rows), 4) if rows else None,
            "oracle_headroom_over_dense": round(
                mean([r["oracle"] - r["dense"] for r in rows]), 4),
            "mean_delta_when_bm25_wins": round(
                mean([r["delta"] for r in bm25_wins]), 4) if bm25_wins else None,
        },
        "D3_tau": quartile_report(rows, n_boot, seed + 10),
        "by_corpus_size": {},
        "by_language": {},
    }
    # DeepSWE repositories are far smaller than the main paper's construction
    # repositories, and a small corpus gives BM25 less to be confused by. This
    # slice separates "the premium does not transfer" from "the premium needs
    # a large corpus to appear".
    sized = [r for r in rows if r.get("n_corpus_files")]
    if len(sized) >= 12:
        sized.sort(key=lambda r: r["n_corpus_files"])
        third = len(sized) // 3
        for label, chunk in (
            ("small", sized[:third]),
            ("medium", sized[third:2 * third]),
            ("large", sized[2 * third:]),
        ):
            point, lo, hi = cluster_bootstrap(by_repo(chunk, "delta"), mean, n_boot, seed + 30)
            report["by_corpus_size"][label] = {
                "n": len(chunk),
                "files_range": [chunk[0]["n_corpus_files"], chunk[-1]["n_corpus_files"]],
                "median_files": chunk[len(chunk) // 2]["n_corpus_files"],
                "mean_delta": round(point, 4),
                "ci": [round(lo, 4), round(hi, 4)],
                "bm25_win_rate": round(
                    mean([1.0 if r["bm25"] > r["dense"] else 0.0 for r in chunk]), 4),
            }
    langs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        langs[row["language"]].append(row)
    for lang, subset in sorted(langs.items()):
        point, lo, hi = cluster_bootstrap(by_repo(subset, "delta"), mean, n_boot, seed + 20)
        report["by_language"][lang] = {
            "n": len(subset),
            "mean_delta": round(point, 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "bm25_win_rate": round(
                mean([1.0 if r["bm25"] > r["dense"] else 0.0 for r in subset]), 4),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    manifest = {row["task_id"]: row for row in read_jsonl(args.manifest)}
    result: dict[str, Any] = {
        "n_boot": args.n_boot, "seed": args.seed,
        "bootstrap": "repository-clustered percentile", "encoders": {},
    }
    per_encoder: dict[str, list[dict[str, Any]]] = {}
    for encoder_dir in sorted(p for p in args.retrieval_root.iterdir() if p.is_dir()):
        rows = load_encoder(encoder_dir, manifest)
        if not rows:
            continue
        per_encoder[encoder_dir.name] = rows
        result["encoders"][encoder_dir.name] = encoder_report(
            encoder_dir.name, rows, args.n_boot, args.seed
        )

    # Cross-encoder agreement on the lexical-rescue population: if the same
    # queries favour BM25 under both encoders, the rescue set is a property of
    # the query, not of one encoder.
    names = sorted(per_encoder)
    if len(names) == 2:
        a, b = (
            {r["task_id"] for r in per_encoder[n] if r["bm25"] > r["dense"]}
            for n in names
        )
        union = a | b
        result["lexical_rescue_agreement"] = {
            "encoders": names,
            f"bm25_wins_{names[0]}": len(a),
            f"bm25_wins_{names[1]}": len(b),
            "intersection": len(a & b),
            "jaccard": round(len(a & b) / len(union), 4) if union else None,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
