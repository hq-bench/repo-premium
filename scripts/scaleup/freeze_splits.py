#!/usr/bin/env python3
"""Freeze deterministic language-stratified repository-disjoint splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_EXTS = {
    "go": ".go",
    "java": ".java",
    "python": ".py,.pyi",
    "rust": ".rs",
    "typescript_javascript": ".ts,.tsx,.js,.jsx,.mjs,.cjs",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def stable_key(repo: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{repo}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--dev-fraction", type=float, default=1 / 3)
    args = parser.parse_args()

    labels = read_jsonl(args.labels)
    manifest = read_jsonl(args.manifest)
    label_by_id = {row["task_id"]: row for row in labels}
    manifest_by_id = {row["task_id"]: row for row in manifest}
    if len(label_by_id) != len(labels) or len(manifest_by_id) != len(manifest):
        raise RuntimeError("duplicate task IDs in labels or manifest")
    if set(label_by_id) != set(manifest_by_id):
        raise RuntimeError("label and manifest task IDs differ")

    repo_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        repo_rows[row["repo"]].append(row)
    by_language: dict[str, list[str]] = defaultdict(list)
    for repo, rows in repo_rows.items():
        languages = {row["language_group"] for row in rows}
        windows = Counter(row["date_window"] for row in rows)
        if len(rows) != 10 or windows != {"pre": 5, "post": 5}:
            raise RuntimeError(f"unbalanced repository cell: {repo} {dict(windows)}")
        if len(languages) != 1:
            raise RuntimeError(f"repository spans language groups: {repo}")
        by_language[next(iter(languages))].append(repo)

    assignment: dict[str, str] = {}
    for language, repos in sorted(by_language.items()):
        ordered = sorted(repos, key=lambda repo: stable_key(repo, args.seed))
        n_dev = max(1, min(len(ordered) - 1, round(len(ordered) * args.dev_fraction)))
        for repo in ordered[:n_dev]:
            assignment[repo] = "dev"
        for repo in ordered[n_dev:]:
            assignment[repo] = "test"

    retrieval: list[dict[str, Any]] = []
    for task_id in sorted(label_by_id):
        label = label_by_id[task_id]
        source = manifest_by_id[task_id]
        retrieval.append({
            **label,
            "repository_url": source["repository_url"],
            "split": assignment[label["repo"]],
            "code_exts": DEFAULT_EXTS[label["language_group"]],
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    assignment_rows = [
        {
            "repo": repo,
            "language_group": repo_rows[repo][0]["language_group"],
            "split": split,
        }
        for repo, split in sorted(assignment.items())
    ]
    write_jsonl(args.out_dir / "repository_splits.jsonl", assignment_rows)
    write_jsonl(args.out_dir / "retrieval_tasks.jsonl", retrieval)
    for split in ("dev", "test"):
        write_jsonl(
            args.out_dir / f"retrieval_tasks_{split}.jsonl",
            [row for row in retrieval if row["split"] == split],
        )

    summary = {
        "schema_version": "coreb-scaleup-splits-v1",
        "seed": args.seed,
        "dev_fraction_target": args.dev_fraction,
        "n_repositories": len(repo_rows),
        "n_tasks": len(retrieval),
        "repositories_by_split": dict(Counter(assignment.values())),
        "tasks_by_split": dict(Counter(row["split"] for row in retrieval)),
        "repositories_by_language_split": {
            language: dict(Counter(assignment[repo] for repo in repos))
            for language, repos in sorted(by_language.items())
        },
        "tasks_by_window_split": {
            split: dict(Counter(row["date_window"] for row in retrieval if row["split"] == split))
            for split in ("dev", "test")
        },
        "repository_disjoint": not (
            {row["repo"] for row in retrieval if row["split"] == "dev"}
            & {row["repo"] for row in retrieval if row["split"] == "test"}
        ),
    }
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
