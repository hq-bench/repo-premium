#!/usr/bin/env python3
"""Prove `git ls-tree` corpus enumeration matches worktree `rglob` enumeration.

The scale-up audit originally enumerated each task corpus by walking a
materialized worktree. Materializing 360 full worktrees on JuiceFS proved
prohibitively slow, so the audit is being switched to read the same file list
straight out of the bare mirror with `git ls-tree`. The two enumerations are
only equivalent if a checkout carries no untracked, ignored, or symlinked
files that `rglob` would see and `ls-tree` would not.

This script checks that claim on every task that currently has BOTH a valid
worktree and a mirror, and fails closed if any task disagrees.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

# Git tree entry modes that materialize as a readable regular file.
BLOB_MODES = {"100644", "100755"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_exts(raw: str) -> tuple[str, ...]:
    return tuple(
        part.strip() if part.strip().startswith(".") else f".{part.strip()}"
        for part in raw.split(",")
        if part.strip()
    )


def rglob_corpus(clone_dir: Path, exts: tuple[str, ...]) -> set[str]:
    """Exactly the enumeration used by audit_corpora.list_corpus."""
    return {
        path.relative_to(clone_dir).as_posix()
        for ext in exts
        for path in clone_dir.rglob(f"*{ext}")
    }


def git(*args: str, cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        ("git",) + args, cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    return proc.returncode, proc.stdout


def lstree_corpus(
    mirror: Path, commit: str, exts: tuple[str, ...]
) -> tuple[set[str], set[str]]:
    """Return (blob paths matching exts, non-blob paths matching exts).

    Non-blob entries are symlinks and submodule gitlinks. They are reported
    separately because they are exactly the cases where a worktree walk and a
    tree listing can legitimately disagree.
    """
    code, out = git("ls-tree", "-r", "-z", commit, cwd=mirror)
    if code != 0:
        raise RuntimeError(f"ls-tree failed in {mirror} at {commit}: {out.strip()[:400]}")
    blobs: set[str] = set()
    others: set[str] = set()
    for record in out.split("\0"):
        if not record:
            continue
        meta, _, path = record.partition("\t")
        if not path:
            continue
        mode = meta.split(" ", 1)[0]
        if not path.endswith(exts):
            continue
        (blobs if mode in BLOB_MODES else others).add(path)
    return blobs, others


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--examples", type=int, default=5,
        help="How many differing paths to record per mismatching task.",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    n_compared = n_match = n_mismatch = n_skipped = 0

    for task in read_jsonl(args.tasks):
        task_id = task["task_id"]
        commit = task["base_commit"]
        exts = normalize_exts(task["code_exts"])
        checkout = args.repo_root / task_id
        mirror = args.mirror_root / f"{task['repo'].replace('/', '__')}.git"

        if not checkout.is_dir() or not mirror.is_dir():
            n_skipped += 1
            continue
        code, head = git("rev-parse", "HEAD", cwd=checkout)
        head = head.strip()
        if code != 0 or head != commit:
            # Partial or interrupted checkout: nothing trustworthy to compare.
            n_skipped += 1
            continue

        try:
            tree, non_blobs = lstree_corpus(mirror, commit, exts)
        except RuntimeError as exc:
            rows.append({
                "task_id": task_id, "status": "lstree_error", "detail": str(exc),
            })
            n_mismatch += 1
            continue

        walk = rglob_corpus(checkout, exts)
        only_walk = sorted(walk - tree)
        only_tree = sorted(tree - walk)
        agree = not only_walk and not only_tree
        n_compared += 1
        n_match += agree
        n_mismatch += not agree

        rows.append({
            "task_id": task_id,
            "repo": task["repo"],
            "status": "match" if agree else "mismatch",
            "n_walk": len(walk),
            "n_tree": len(tree),
            "n_only_walk": len(only_walk),
            "n_only_tree": len(only_tree),
            "n_non_blob_entries": len(non_blobs),
            "example_only_walk": only_walk[: args.examples],
            "example_only_tree": only_tree[: args.examples],
            "example_non_blob": sorted(non_blobs)[: args.examples],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {
        "n_tasks_in_manifest": len(read_jsonl(args.tasks)),
        "n_compared": n_compared,
        "n_match": n_match,
        "n_mismatch": n_mismatch,
        "n_skipped_no_valid_checkout": n_skipped,
        "n_tasks_with_non_blob_entries": sum(
            1 for row in rows if row.get("n_non_blob_entries")
        ),
        "detail": str(args.out),
    }
    print(json.dumps(summary, indent=2))
    return int(n_mismatch > 0 or n_compared == 0)


if __name__ == "__main__":
    raise SystemExit(main())
