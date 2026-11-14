#!/usr/bin/env python3
"""Characterise the queries on which BM25 beats dense retrieval.

The paper reports *how often* lexical retrieval rescues dense retrieval. This
script answers *which queries* it rescues, using measurable, reproducible
properties of the query text rather than model-assisted labelling, so the
taxonomy can be recomputed and audited.

For every query it extracts structural evidence features:

  path_literal      a file path with a code extension appears in the query
  gold_basename     a gold file's basename appears verbatim in the query
  traceback         a Python/JS traceback or exception-type marker appears
  error_string      a quoted message or `Error:`/`Warning:`-style token
  code_block        a fenced code block is present
  dotted_call       a qualified call like `mod.Class.method(` is present
  rare_identifier   a camelCase/snake_case identifier shared with the gold file
  version_literal   a version or release literal such as 3.11 or v2.4.1
  numeric_id        a bare issue/PR reference such as #19911

The informative output is not the raw prevalence of a feature but its
prevalence among BM25-win queries *relative to* dense-win queries: a feature
that is equally common in both explains nothing.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

CODE_EXT = r"(?:py|pyi|js|jsx|ts|tsx|go|rs|java|kt|scala|c|h|cpp|rb|cfg|toml|ini)"

PATTERNS: dict[str, re.Pattern[str]] = {
    "path_literal": re.compile(rf"\b[\w./-]+/[\w.-]+\.{CODE_EXT}\b"),
    "traceback": re.compile(
        r"Traceback \(most recent call last\)|^\s*File \"|"
        r"\b\w*(?:Error|Exception)\b\s*:", re.MULTILINE),
    "error_string": re.compile(
        r"`[^`\n]*(?:error|warning|failed|cannot|invalid|unexpected)[^`\n]*`"
        r"|\"[^\"\n]*(?:Error|Warning)[^\"\n]*\"", re.IGNORECASE),
    "code_block": re.compile(r"```"),
    "dotted_call": re.compile(r"\b\w+(?:\.\w+){1,}\s*\("),
    "version_literal": re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b"),
    "numeric_id": re.compile(r"#\d{2,}"),
}

IDENT = re.compile(r"\b(?:[a-z]+(?:_[a-z0-9]+)+|[a-z]+(?:[A-Z][a-z0-9]+)+)\b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def features(query: str, gold_files: list[str]) -> dict[str, bool]:
    found = {name: bool(rx.search(query)) for name, rx in PATTERNS.items()}
    basenames = {Path(g).name for g in gold_files}
    stems = {Path(g).stem for g in gold_files}
    found["gold_basename"] = any(b in query for b in basenames)
    # An identifier shared between query and gold *stem* is the strongest
    # lexical anchor BM25 can exploit, and is what tau formalises.
    query_idents = set(IDENT.findall(query))
    found["rare_identifier"] = bool(query_idents & stems)
    return found


def collect(results_root: Path, qrel_dir: Path, cell: str) -> list[dict[str, Any]]:
    qrels: dict[str, dict[str, Any]] = {}
    for path in sorted(qrel_dir.glob("*.jsonl")):
        if path.name.startswith("._"):
            continue
        for row in read_jsonl(path):
            qrels[row["query_id"]] = row

    rows: list[dict[str, Any]] = []
    for repo_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        record_path = repo_dir / "smoke_test_django.jsonl"
        if not record_path.is_file():
            continue
        for record in read_jsonl(record_path):
            meta = qrels.get(record["query_id"])
            if meta is None:
                continue
            query = meta.get("problem_statement") or ""
            dense = record.get("dense_ndcg10", 0.0)
            bm25 = record.get("bm25_ndcg10", 0.0)
            rows.append({
                "cell": cell,
                "query_id": record["query_id"],
                "repo": repo_dir.name,
                "tau": record.get("tau_q"),
                "dense": dense,
                "bm25": bm25,
                "delta": dense - bm25,
                "winner": "bm25" if bm25 > dense else ("dense" if dense > bm25 else "tie"),
                "query_chars": len(query),
                **features(query, meta.get("gold_files", [])),
            })
    return rows


def summarise(rows: list[dict[str, Any]], cell: str) -> dict[str, Any]:
    bm25_win = [r for r in rows if r["winner"] == "bm25"]
    dense_win = [r for r in rows if r["winner"] == "dense"]
    ties = [r for r in rows if r["winner"] == "tie"]

    table: dict[str, Any] = {}
    for name in list(PATTERNS) + ["gold_basename", "rare_identifier"]:
        in_bm25 = sum(r[name] for r in bm25_win)
        in_dense = sum(r[name] for r in dense_win)
        p_bm25 = in_bm25 / len(bm25_win) if bm25_win else 0.0
        p_dense = in_dense / len(dense_win) if dense_win else 0.0
        table[name] = {
            "prevalence_bm25_win": round(p_bm25, 4),
            "prevalence_dense_win": round(p_dense, 4),
            "lift": round(p_bm25 / p_dense, 3) if p_dense else None,
            "n_bm25_win": in_bm25,
            "n_dense_win": in_dense,
        }

    def mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else float("nan")

    return {
        "cell": cell,
        "n_queries": len(rows),
        "n_bm25_win": len(bm25_win),
        "n_dense_win": len(dense_win),
        "n_tie": len(ties),
        "bm25_win_rate": round(len(bm25_win) / len(rows), 4) if rows else None,
        "mean_tau": {
            "bm25_win": mean([r["tau"] for r in bm25_win if r["tau"] is not None]),
            "dense_win": mean([r["tau"] for r in dense_win if r["tau"] is not None]),
        },
        "mean_query_chars": {
            "bm25_win": mean([float(r["query_chars"]) for r in bm25_win]),
            "dense_win": mean([float(r["query_chars"]) for r in dense_win]),
        },
        "features_sorted_by_lift": dict(sorted(
            table.items(),
            key=lambda kv: (kv[1]["lift"] is None, -(kv[1]["lift"] or 0)),
        )),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append", nargs=3,
                        metavar=("NAME", "RESULTS_DIR", "QREL_DIR"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {"cells": {}}
    for name, results_dir, qrel_dir in args.cell:
        rows = collect(Path(results_dir), Path(qrel_dir), name)
        all_rows.extend(rows)
        report["cells"][name] = summarise(rows, name)
    report["pooled"] = summarise(all_rows, "pooled")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    (args.out.parent / "lexical_rescue_per_query.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in all_rows)
    )

    for cell, rep in list(report["cells"].items()) + [("pooled", report["pooled"])]:
        print(f"\n=== {cell} === n={rep['n_queries']} "
              f"bm25_win={rep['n_bm25_win']} dense_win={rep['n_dense_win']} "
              f"tie={rep['n_tie']} rate={rep['bm25_win_rate']}")
        print(f"  mean tau  bm25-win {rep['mean_tau']['bm25_win']} "
              f"vs dense-win {rep['mean_tau']['dense_win']}")
        print(f"  {'feature':<18}{'BM25-win':>10}{'dense-win':>11}{'lift':>8}")
        for name, stats in rep["features_sorted_by_lift"].items():
            lift = "n/a" if stats["lift"] is None else f"{stats['lift']:.2f}"
            print(f"  {name:<18}{stats['prevalence_bm25_win']:>10.3f}"
                  f"{stats['prevalence_dense_win']:>11.3f}{lift:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
