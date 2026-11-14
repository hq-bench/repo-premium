#!/usr/bin/env python3
"""Deterministically replace tasks whose production-code gold is not distinct.

Main qrels are defined to hold base-visible *production-code* files. A task
whose distinguishing gold is a test fixture therefore loses that gold, and two
tasks in the same repository/window can collapse onto an identical gold set.
Such a collapse makes them indistinguishable as retrieval targets even though
they are different pull requests.

This tool applies the production-code rule, finds identical-gold collisions
within each (repository, date window) cell, keeps the earlier task under the
original selection order, and draws replacements for the rest from the same
cell's candidate pool using that same order. Replacements must have non-empty
gold, every gold path inside the repository's retrieval corpus extensions, and
a gold set distinct from everything retained or already drawn.

Selection order is `sha256(f"{seed}:{task_id}")`, identical to
`select_balanced.py`, so the result is reproducible and pool position is not
re-randomized by this pass.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_order(task_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def production_gold(
    gold: list[str], corpus_exts: tuple[str, ...], exclude: tuple[str, ...]
) -> tuple[str, ...]:
    """Gold restricted to files the retrieval corpus can actually contain."""
    kept = [
        path for path in gold
        if path.endswith(corpus_exts)
        and not any(fnmatch.fnmatch(path, pattern) for pattern in exclude)
    ]
    return tuple(sorted(kept))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True,
                        help="Frozen retrieval_tasks.jsonl.")
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--repo", action="append", default=[],
                        help="Restrict to these repositories (repeatable).")
    parser.add_argument("--window", action="append", default=[],
                        help="Restrict to these date windows (repeatable).")
    parser.add_argument("--exclude-glob", action="append", default=[],
                        help="Gold path globs treated as non-production.")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tasks = read_jsonl(args.tasks)
    exclude = tuple(args.exclude_glob)
    repo_filter = set(args.repo)
    window_filter = set(args.window)

    # Corpus extensions are a property of the repository, not of one task.
    corpus_exts: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        exts = tuple(
            part.strip() if part.strip().startswith(".") else f".{part.strip()}"
            for part in task["code_exts"].split(",") if part.strip()
        )
        corpus_exts.setdefault(task["repo"], exts)

    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_cell[(task["repo"], task["date_window"])].append(task)

    pool: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    selected_ids = {task["task_id"] for task in tasks}
    for path in args.candidates:
        for row in read_jsonl(path):
            if not row.get("eligible_metadata"):
                continue
            window = row.get("date_window")
            if window is None:
                window = "pre" if "pre" in path.name or "pre" in str(path.parent) else "post"
            pool[(row["repo"], window)].append(row)

    plan: list[dict[str, Any]] = []
    for (repo, window), cell in sorted(by_cell.items()):
        if repo_filter and repo not in repo_filter:
            continue
        if window_filter and window not in window_filter:
            continue
        exts = corpus_exts[repo]
        ordered = sorted(cell, key=lambda t: stable_order(t["task_id"], args.seed))

        keep: dict[tuple[str, ...], str] = {}
        needs_replacement: list[dict[str, Any]] = []
        for task in ordered:
            gold = production_gold(task["main_gold_files"], exts, exclude)
            if not gold:
                needs_replacement.append({"task": task, "reason": "no production gold in corpus"})
            elif gold in keep:
                needs_replacement.append({
                    "task": task,
                    "reason": f"identical production gold to {keep[gold]}",
                })
            else:
                keep[gold] = task["task_id"]

        if not needs_replacement:
            continue

        taken = set(keep)
        candidates = sorted(
            (row for row in pool[(repo, window)] if row["task_id"] not in selected_ids),
            key=lambda r: stable_order(r["task_id"], args.seed),
        )
        drawn: list[dict[str, Any]] = []
        for cand in candidates:
            if len(drawn) == len(needs_replacement):
                break
            gold = production_gold(cand["main_gold_files"], exts, exclude)
            if not gold or gold in taken:
                continue
            taken.add(gold)
            drawn.append(cand)

        for index, item in enumerate(needs_replacement):
            replacement = drawn[index] if index < len(drawn) else None
            plan.append({
                "repo": repo,
                "date_window": window,
                "replace_task_id": item["task"]["task_id"],
                "reason": item["reason"],
                "original_gold": item["task"]["main_gold_files"],
                "original_production_gold": list(
                    production_gold(item["task"]["main_gold_files"], exts, exclude)
                ),
                "replacement_task_id": replacement["task_id"] if replacement else None,
                "replacement_pr_number": replacement["pr_number"] if replacement else None,
                "replacement_title": replacement["title"] if replacement else None,
                "replacement_base_commit": replacement["base_commit"] if replacement else None,
                "replacement_gold": replacement["main_gold_files"] if replacement else None,
                "status": "resolved" if replacement else "no_eligible_candidate",
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in plan))
    unresolved = [row for row in plan if row["status"] != "resolved"]
    summary = {
        "seed": args.seed,
        "exclude_globs": list(exclude),
        "n_replacements_needed": len(plan),
        "n_resolved": len(plan) - len(unresolved),
        "n_unresolved": len(unresolved),
        "plan": str(args.out),
    }
    print(json.dumps(summary, indent=2))
    for row in plan:
        print(
            f"  {row['replace_task_id']} -> {row['replacement_task_id']}"
            f"  ({row['reason']})"
        )
    return int(bool(unresolved))


if __name__ == "__main__":
    raise SystemExit(main())
