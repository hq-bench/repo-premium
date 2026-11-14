#!/usr/bin/env python3
"""The inverse intervention: does removing lexical anchors restore the premium?

`sec:genarm` shows that *adding* model-generated identifier anchors to a query
removes the dense-over-BM25 premium. If the premium is a property of the query
text rather than of code search, the mirror manipulation must also hold:
*stripping* the anchors out of a query should push the premium back up.

Cell C already supplies that manipulation. Each Cell C query is an LLM rewrite
of a Cell A or Cell B query into behavior-only prose with every identifier,
path, and error string removed, and it inherits its source query's gold files.
So each rewrite is paired to its source at the same gold, over the same corpus,
and the only thing that changed is the query text.

For each query we form the double difference

    premium(q)          = nDCG@10_dense(q) - nDCG@10_bm25(q)
    inverse_effect(q)   = premium(rewritten q) - premium(source q)

and predict inverse_effect > 0. Confidence intervals are a repository-clustered
paired bootstrap: the resampling unit is the repository, because queries inside
one repository share a corpus and an IDF profile.

CAVEAT, stated here because the number is easy to over-read: the rewrite also
shortens and paraphrases. A positive inverse effect is therefore consistent
with anchor removal *and* with a generic paraphrase penalty. Isolating anchor
removal needs a second arm that paraphrases while PRESERVING identifiers; that
arm does not exist yet. Report this as directional evidence until it does.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

# Cell C run directory -> canonical source-run prefix in canon202607/.
ENCODERS = {
    "jina-0.5b": "jina05",
    "jina-1.5b": "jina15",
    "Qwen3-0.6B": "qwen06",
    "Qwen3-4B": "qwen4",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(root: Path) -> dict[str, dict[str, Any]]:
    """Map query_id -> record over every repository directory in a run."""
    out: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return out
    for repo_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        record_path = repo_dir / "smoke_test_django.jsonl"
        if not record_path.is_file():
            continue
        for record in read_jsonl(record_path):
            record["_repo"] = repo_dir.name
            out[record["query_id"]] = record
    return out


def premium(record: dict[str, Any]) -> float:
    return record["dense_ndcg10"] - record["bm25_ndcg10"]


def clustered_bootstrap(
    by_repo: dict[str, list[float]], n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Resample repositories with replacement, then queries within each draw."""
    repos = list(by_repo)
    flat = [v for vs in by_repo.values() for v in vs]
    if not flat:
        return float("nan"), float("nan"), float("nan")
    point = sum(flat) / len(flat)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in range(len(repos)):
            values = by_repo[repos[rng.randrange(len(repos))]]
            sample.extend(values[rng.randrange(len(values))] for _ in values)
        draws.append(sum(sample) / len(sample))
    draws.sort()
    return point, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]


def summarize(rows: list[dict[str, Any]], key: str, n_boot: int, seed: int) -> dict[str, Any]:
    by_repo: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_repo[row["repo"]].append(row[key])
    point, lo, hi = clustered_bootstrap(by_repo, n_boot, seed)
    return {
        "mean": round(point, 4),
        "ci": [round(lo, 4), round(hi, 4)],
        "n": len(rows),
        "n_repositories": len(by_repo),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True,
                        help="smoke_test_results/ containing cellC_* and canon202607/")
    parser.add_argument("--canon-subdir", default="canon202607")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "design": "paired double difference; Cell C rewrite minus its source query",
        "prediction": "inverse_effect > 0 (stripping anchors restores the premium)",
        "bootstrap": "repository-clustered paired, resampling repositories then queries",
        "n_boot": args.n_boot,
        "caveat": (
            "the rewrite also shortens and paraphrases; an identifier-preserving "
            "paraphrase control is required to attribute this to anchor removal alone"
        ),
        "by_encoder": {},
    }

    pooled_rows: list[dict[str, Any]] = []
    for cellc_name, canon_prefix in ENCODERS.items():
        cellc = load_run(args.results_root / f"cellC_{cellc_name}_c1000")
        source: dict[str, dict[str, Any]] = {}
        for cell in ("cellA", "cellB"):
            run = load_run(args.results_root / args.canon_subdir / f"{canon_prefix}_{cell}_c1000")
            for qid, record in run.items():
                record["_cell"] = cell
                source[qid] = record
        if not cellc:
            report["by_encoder"][cellc_name] = {"error": "no Cell C run found"}
            continue

        rows: list[dict[str, Any]] = []
        unmatched: list[str] = []
        for qid, rewritten in cellc.items():
            src_id = qid[:-2] if qid.endswith("_c") else qid
            src = source.get(src_id)
            if src is None:
                unmatched.append(qid)
                continue
            rows.append({
                "query_id": qid,
                "source_query_id": src_id,
                "repo": rewritten["_repo"],
                "source_cell": src["_cell"],
                "premium_source": premium(src),
                "premium_rewritten": premium(rewritten),
                "inverse_effect": premium(rewritten) - premium(src),
                "tau_source": src.get("tau_q"),
                "tau_rewritten": rewritten.get("tau_q"),
                "tau_shift": (rewritten.get("tau_q") or 0.0) - (src.get("tau_q") or 0.0),
                "dense_delta": rewritten["dense_ndcg10"] - src["dense_ndcg10"],
                "bm25_delta": rewritten["bm25_ndcg10"] - src["bm25_ndcg10"],
            })

        entry: dict[str, Any] = {
            "n_paired": len(rows),
            "n_unmatched": len(unmatched),
            "inverse_effect": summarize(rows, "inverse_effect", args.n_boot, args.seed),
            "premium_source": summarize(rows, "premium_source", args.n_boot, args.seed),
            "premium_rewritten": summarize(rows, "premium_rewritten", args.n_boot, args.seed),
            "tau_shift": summarize(rows, "tau_shift", args.n_boot, args.seed),
            # Which side moves: the mechanism predicts BM25 falls further than dense.
            "dense_delta": summarize(rows, "dense_delta", args.n_boot, args.seed),
            "bm25_delta": summarize(rows, "bm25_delta", args.n_boot, args.seed),
            "by_source_cell": {},
        }
        for cell in ("cellA", "cellB"):
            subset = [r for r in rows if r["source_cell"] == cell]
            if subset:
                entry["by_source_cell"][cell] = summarize(
                    subset, "inverse_effect", args.n_boot, args.seed
                )
        if unmatched:
            entry["unmatched_examples"] = unmatched[:5]
        report["by_encoder"][cellc_name] = entry

        for row in rows:
            pooled_rows.append({**row, "encoder": cellc_name})

    if pooled_rows:
        report["pooled_over_encoders"] = {
            "inverse_effect": summarize(pooled_rows, "inverse_effect", args.n_boot, args.seed),
            "n_encoders": len({r["encoder"] for r in pooled_rows}),
            "note": "encoders share the same 125 queries, so this is not an independent sample",
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
