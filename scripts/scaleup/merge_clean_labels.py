#!/usr/bin/env python3
"""Merge retained adjudications and verified replacements into clean labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPLACEMENT_FULL_RAW_REVIEWED = {
    "axios__axios-pr-5831",
    "axios__axios-pr-6091",
    "elastic__elasticsearch-pr-156180",
    "microsoft__TypeScript-pr-63401",
    "python-poetry__poetry-pr-9168",
    "rust-lang__cargo-pr-13619",
    "tauri-apps__tauri-pr-9151",
}


def rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]


def validate_clean_rows(
    clean: list[dict[str, Any]], manifest: list[dict[str, Any]]
) -> None:
    clean_ids = [row["task_id"] for row in clean]
    manifest_ids = [row["task_id"] for row in manifest]
    if len(clean_ids) != len(set(clean_ids)):
        raise RuntimeError("clean labels contain duplicate task IDs")
    if set(clean_ids) != set(manifest_ids):
        raise RuntimeError("clean-label task IDs do not match the clean manifest")
    if any(not row["raw_query"].strip() for row in clean):
        raise RuntimeError("clean labels contain an empty raw query")
    if any(not row["curated_query"].strip() for row in clean):
        raise RuntimeError("clean labels contain an empty curated query")
    if any(not row["main_gold_files"] for row in clean):
        raise RuntimeError("clean labels contain an empty main-gold set")
    if any(row["label_quality"]["human_verified"] for row in clean):
        raise RuntimeError("assistant/model labels must not claim human verification")
    if any(
        row["label_quality"]["assistant_decision"] == "replace" for row in clean
    ):
        raise RuntimeError("a rejected label remained in the clean output")
    replacement_ids = {
        row["task_id"] for row in clean if row["provenance"]["replaces_task_ids"]
    }
    if replacement_ids != REPLACEMENT_FULL_RAW_REVIEWED:
        raise RuntimeError(
            "clean replacement provenance does not match reviewed replacements"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path, required=True)
    parser.add_argument("--replacement-labels", type=Path, required=True)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = rows(args.manifest)
    original = {row["task_id"]: row for row in rows(args.original_manifest)}
    adjudicated = {row["task_id"]: row for row in rows(args.adjudicated)}
    replacements = {row["task_id"]: row for row in rows(args.replacement_labels)}
    removed_by_cell: dict[tuple[str, str], list[str]] = defaultdict(list)
    for task_id, row in adjudicated.items():
        if row["assistant_adjudication"]["action"] == "replace":
            source = original[task_id]
            removed_by_cell[(source["repo"], source["date_window"])].append(task_id)

    clean: list[dict[str, Any]] = []
    for task in manifest:
        task_id = task["task_id"]
        if task_id in adjudicated:
            source = adjudicated[task_id]
            action = source["assistant_adjudication"]["action"]
            if action == "replace":
                raise RuntimeError(f"rejected task remained in clean manifest: {task_id}")
            curated = source["assistant_curated_query"]
            review_scope = source["assistant_adjudication"]["review_scope"]
            decision = source["assistant_adjudication"]["decision"]
            label_origin = "assistant_revision" if action == "revise" else "qwen_v2"
            qwen_verified = bool(source["llm_verified"])
            source_underspecified = bool(source["assistant_source_underspecified"])
            provenance = {"replaces_task_ids": []}
        elif task_id in replacements:
            source = replacements[task_id]
            if task_id not in REPLACEMENT_FULL_RAW_REVIEWED:
                raise RuntimeError(f"replacement lacks full-raw review: {task_id}")
            curated = source["curated_query"]
            review_scope = "full_raw"
            decision = "pass"
            label_origin = "replacement_qwen_v2"
            qwen_verified = bool(source["llm_verified"])
            source_underspecified = bool(
                source["final_audit"].get("source_underspecified", False)
            )
            provenance = {
                "replaces_task_ids": sorted(
                    removed_by_cell[(task["repo"], task["date_window"])]
                )
            }
        else:
            raise RuntimeError(f"no label source for clean task: {task_id}")

        clean.append(
            {
                "task_id": task_id,
                "repo": task["repo"],
                "language_group": task["language_group"],
                "date_window": task["date_window"],
                "base_commit": task["base_commit"],
                "merged_at": task["merged_at"],
                "raw_query": task["raw_query"],
                "curated_query": curated,
                "main_gold_files": task["main_gold_files"],
                "reference_loc": task["reference_loc"],
                "label_quality": {
                    "schema_version": "coreb-scaleup-clean-v1",
                    "label_origin": label_origin,
                    "qwen_model": "qwen35-122B",
                    "qwen_prompt_version": "coreb-scaleup-curation-v2",
                    "qwen_verified": qwen_verified,
                    "assistant_review_scope": review_scope,
                    "assistant_decision": decision,
                    "source_underspecified": source_underspecified,
                    "human_verified": False,
                },
                "provenance": provenance,
            }
        )

    clean.sort(key=lambda row: (row["repo"], row["date_window"], row["task_id"]))
    validate_clean_rows(clean, manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "coreb_scaleup_clean_labels_v1.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in clean)
    )
    summary = {
        "schema_version": "coreb-scaleup-clean-v1",
        "n_tasks": len(clean),
        "n_repositories": len({row["repo"] for row in clean}),
        "counts_by_window": dict(Counter(row["date_window"] for row in clean)),
        "counts_by_language": dict(
            Counter(row["language_group"] for row in clean)
        ),
        "counts_by_label_origin": dict(
            Counter(row["label_quality"]["label_origin"] for row in clean)
        ),
        "counts_by_review_scope": dict(
            Counter(row["label_quality"]["assistant_review_scope"] for row in clean)
        ),
        "n_qwen_verified": sum(
            row["label_quality"]["qwen_verified"] for row in clean
        ),
        "n_source_underspecified": sum(
            row["label_quality"]["source_underspecified"] for row in clean
        ),
        "n_replacements": sum(bool(row["provenance"]["replaces_task_ids"]) for row in clean),
        "human_verification_complete": False,
    }
    (args.out_dir / "coreb_scaleup_clean_labels_v1_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return int(len(clean) != 360 or summary["n_repositories"] != 36)


if __name__ == "__main__":
    raise SystemExit(main())
