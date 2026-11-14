#!/usr/bin/env python3
"""Prepare exact-base task worktrees with one shared object cache per repository."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def run(*args: str, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        args, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if proc.returncode:
        detail = proc.stdout.strip()[-4000:] or "no command output"
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(args)}\n{detail}")
    return proc.stdout.strip()


def exact_clean_checkout(target: Path, commit: str) -> bool:
    if not target.exists():
        return False
    try:
        head = run("git", "rev-parse", "HEAD", cwd=target)
        dirty = run("git", "status", "--porcelain", cwd=target)
        return head == commit and not dirty
    except RuntimeError:
        return False


def quarantine(target: Path, quarantine_root: Path) -> Path:
    quarantine_root.mkdir(parents=True, exist_ok=True)
    candidate = quarantine_root / f"{target.name}.{int(time.time())}"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = quarantine_root / f"{target.name}.{int(time.time())}.{suffix}"
    target.rename(candidate)
    return candidate


def prepare_repository(
    repo: str,
    tasks: list[dict[str, Any]],
    repo_root: Path,
    mirror_root: Path,
    quarantine_root: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for task in tasks:
        target = repo_root / task["task_id"]
        if exact_clean_checkout(target, task["base_commit"]):
            results.append({
                "task_id": task["task_id"], "repo": repo,
                "target": str(target), "status": "ready_existing",
                "detail": task["base_commit"],
            })
        else:
            pending.append(task)
    if not pending:
        return results

    mirror = mirror_root / f"{repo.replace('/', '__')}.git"
    try:
        if not mirror.exists():
            run("git", "init", "--bare", str(mirror))
            run("git", "remote", "add", "origin", tasks[0]["repository_url"], cwd=mirror)
        elif not (mirror / "HEAD").exists():
            raise RuntimeError(f"mirror path is not a bare Git repository: {mirror}")
        commits = sorted({task["base_commit"] for task in pending})
        refspecs = [f"{commit}:refs/coreb/{commit}" for commit in commits]
        run(
            "git", "fetch", "--no-tags", "--depth=1", "origin", *refspecs,
            cwd=mirror,
        )
    except Exception as exc:
        results.extend({
            "task_id": task["task_id"], "repo": repo,
            "target": str(repo_root / task["task_id"]), "status": "failed",
            "detail": f"mirror preparation failed: {exc}",
        } for task in pending)
        return results

    for task in pending:
        target = repo_root / task["task_id"]
        try:
            moved = quarantine(target, quarantine_root) if target.exists() else None
            run(
                "git", "worktree", "add", "--detach", str(target),
                task["base_commit"], cwd=mirror,
            )
            if not exact_clean_checkout(target, task["base_commit"]):
                raise RuntimeError("created worktree failed exact-commit/clean check")
            detail = task["base_commit"]
            if moved:
                detail += f"; interrupted checkout preserved at {moved}"
            results.append({
                "task_id": task["task_id"], "repo": repo,
                "target": str(target), "status": "ready_shared", "detail": detail,
            })
        except Exception as exc:
            results.append({
                "task_id": task["task_id"], "repo": repo,
                "target": str(target), "status": "failed", "detail": str(exc),
            })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    tasks = [
        json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()
    ]
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_repo[task["repo"]].append(task)
    args.repo_root.mkdir(parents=True, exist_ok=True)
    args.mirror_root.mkdir(parents=True, exist_ok=True)
    quarantine_root = args.repo_root.parent / "quarantine_interrupted"

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                prepare_repository, repo, rows, args.repo_root,
                args.mirror_root, quarantine_root,
            ): repo
            for repo, rows in sorted(by_repo.items())
        }
        for future in concurrent.futures.as_completed(futures):
            repo_results = future.result()
            results.extend(repo_results)
            ready = sum(row["status"].startswith("ready") for row in repo_results)
            failed = len(repo_results) - ready
            print(f"[{futures[future]}] ready={ready} failed={failed}", flush=True)
            for row in repo_results:
                if row["status"] == "failed":
                    print(f"  [failed] {row['task_id']}: {row['detail']}", flush=True)

    results.sort(key=lambda row: row["task_id"])
    status_path = args.repo_root.parent / "prepare_shared_status.jsonl"
    status_path.write_text("".join(json.dumps(row) + "\n" for row in results))
    failed = sum(not row["status"].startswith("ready") for row in results)
    print(f"ready={len(results) - failed} failed={failed} status={status_path}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
