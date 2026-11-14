#!/usr/bin/env python3
"""Replace rejected tasks deterministically within repository/date cells."""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from select_balanced import stable_order


BAD_TITLE = re.compile(
    r"(?i)(cherry.?pick|backport|\bpick PR\b|🤖|^\[release\]|"
    r"^release\b|^\[\d+\.\d+\])"
)


def normalized_title(value: str) -> str:
    value = re.sub(
        r"(?i)\b(backport|cherry.?pick|release|stable|fix|feat|chore|docs?|"
        r"refactor|into|from|pr)\b",
        " ",
        value,
    )
    value = re.sub(r"#?\d+|[^a-zA-Z0-9]+", " ", value.lower())
    return " ".join(value.split())


def duplicate_of(candidate: dict[str, Any], retained: list[dict[str, Any]]) -> bool:
    title = normalized_title(candidate["title"])
    gold = set(candidate.get("main_gold_files", []))
    for other in retained:
        if other["repo"] != candidate["repo"]:
            continue
        if other["date_window"] != candidate["date_window"]:
            continue
        other_gold = set(other.get("main_gold_files", []))
        union = gold | other_gold
        overlap = len(gold & other_gold) / len(union) if union else 0.0
        similarity = difflib.SequenceMatcher(
            None, title, normalized_title(other["title"])
        ).ratio()
        if similarity >= 0.85 and overlap > 0:
            return True
    return False


def low_quality(candidate: dict[str, Any]) -> str | None:
    if BAD_TITLE.search(candidate["title"]):
        return "release_or_backport_title"
    gold = candidate.get("main_gold_files", [])
    if any(path.startswith("dist/") or ".min." in path for path in gold):
        return "generated_distribution_gold"
    return None


def load_candidates(paths: list[Path], window: str) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("eligible_metadata"):
                row["date_window"] = window
                by_id[row["task_id"]] = row
    return list(by_id.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path, required=True)
    parser.add_argument("--pre", type=Path, nargs="+", required=True)
    parser.add_argument("--post", type=Path, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    selected = [
        json.loads(line)
        for line in args.selected.read_text().splitlines()
        if line.strip()
    ]
    adjudicated = {
        row["task_id"]: row
        for row in (
            json.loads(line)
            for line in args.adjudicated.read_text().splitlines()
            if line.strip()
        )
    }
    rejected = {
        task_id
        for task_id, row in adjudicated.items()
        if row["assistant_adjudication"]["action"] == "replace"
    }
    retained = [dict(row) for row in selected if row["task_id"] not in rejected]
    selected_ids = {row["task_id"] for row in selected}
    needs: Counter[tuple[str, str]] = Counter(
        (row["repo"], row["date_window"])
        for row in selected
        if row["task_id"] in rejected
    )
    pool = load_candidates(args.pre, "pre") + load_candidates(args.post, "post")
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_cell[(row["repo"], row["date_window"])].append(row)

    replacements: list[dict[str, Any]] = []
    exclusions: dict[str, list[str]] = defaultdict(list)
    for cell in sorted(needs):
        candidates = sorted(
            by_cell[cell], key=lambda row: stable_order(row, args.seed)
        )
        chosen = 0
        for candidate in candidates:
            task_id = candidate["task_id"]
            if task_id in selected_ids:
                continue
            reason = low_quality(candidate)
            if reason:
                exclusions[task_id].append(reason)
                continue
            if duplicate_of(candidate, retained + replacements):
                exclusions[task_id].append("similar_title_and_overlapping_gold")
                continue
            replacements.append(candidate)
            chosen += 1
            if chosen == needs[cell]:
                break
        if chosen != needs[cell]:
            raise RuntimeError(f"cell {cell} needs {needs[cell]}, found {chosen}")

    clean = retained + replacements
    clean.sort(key=lambda row: (row["repo"], row["date_window"], row["task_id"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "clean_balanced_tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in clean)
    )
    (args.out_dir / "replacement_tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in replacements)
    )
    replacement_rows = []
    for row in replacements:
        replacement_rows.append(
            {
                "task_id": row["task_id"],
                "repo": row["repo"],
                "date_window": row["date_window"],
                "language_group": row["language_group"],
                "title": row["title"],
            }
        )
    summary = {
        "seed": args.seed,
        "n_original": len(selected),
        "n_rejected": len(rejected),
        "n_replacements": len(replacements),
        "n_clean": len(clean),
        "n_repositories": len({row["repo"] for row in clean}),
        "counts_by_window": dict(Counter(row["date_window"] for row in clean)),
        "counts_by_language": dict(
            Counter(row["language_group"] for row in clean)
        ),
        "replacement_rows": replacement_rows,
        "rejected_task_ids": sorted(rejected),
        "candidate_exclusions": dict(exclusions),
    }
    (args.out_dir / "clean_manifest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return int(len(clean) != len(selected))


if __name__ == "__main__":
    raise SystemExit(main())
