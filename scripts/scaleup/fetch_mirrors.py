#!/usr/bin/env python3
"""Fetch every task base commit into one bare mirror per repository.

This is the object-acquisition half of exact-base preparation, split out from
worktree materialization. Fetching writes a small number of large packfiles,
which the shared filesystem handles well; it was never the bottleneck. Nothing
here checks out a working tree.

Safe to re-run: commits already present are skipped, so an interrupted run
resumes where it stopped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def git(*args: str, cwd: Path | None = None, timeout: int | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        ("git",) + args, cwd=cwd, text=True, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    return proc.returncode, proc.stdout


def has_commit(mirror: Path, commit: str) -> bool:
    code, out = git("cat-file", "-t", commit, cwd=mirror)
    return code == 0 and out.strip() == "commit"


def mirror_path(mirror_root: Path, repo: str) -> Path:
    return mirror_root / f"{repo.replace('/', '__')}.git"


def prepare_mirror(mirror: Path, url: str) -> None:
    if not mirror.exists():
        code, out = git("init", "--bare", str(mirror))
        if code != 0:
            raise RuntimeError(f"git init --bare failed: {out.strip()[:400]}")
        code, out = git("remote", "add", "origin", url, cwd=mirror)
        if code != 0:
            raise RuntimeError(f"git remote add failed: {out.strip()[:400]}")
    elif not (mirror / "HEAD").exists():
        raise RuntimeError(f"path exists but is not a bare repository: {mirror}")


def fetch_repo(
    repo: str,
    tasks: list[dict[str, Any]],
    mirror_root: Path,
    attempts: int,
    timeout: int,
) -> dict[str, Any]:
    mirror = mirror_path(mirror_root, repo)
    started = time.monotonic()
    try:
        prepare_mirror(mirror, tasks[0]["repository_url"])
    except RuntimeError as exc:
        return {
            "repo": repo, "status": "failed", "detail": str(exc),
            "n_wanted": len(tasks), "n_present": 0, "n_fetched": 0,
        }

    wanted = sorted({task["base_commit"] for task in tasks})
    present = [commit for commit in wanted if has_commit(mirror, commit)]
    missing = [commit for commit in wanted if commit not in set(present)]
    if not missing:
        return {
            "repo": repo, "status": "complete", "detail": "all commits already present",
            "n_wanted": len(wanted), "n_present": len(present), "n_fetched": 0,
            "seconds": round(time.monotonic() - started, 1),
        }

    # One fetch carrying every missing commit; retry the whole batch on
    # transient network failure rather than leaving a half-fetched mirror.
    refspecs = [f"{commit}:refs/coreb/{commit}" for commit in missing]
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            code, out = git(
                "fetch", "--no-tags", "--depth=1", "origin", *refspecs,
                cwd=mirror, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            code, out = 1, f"fetch exceeded {timeout}s"
        if code == 0:
            break
        last_error = out.strip()[-600:]
        if attempt < attempts:
            time.sleep(min(30, 5 * attempt))
    else:
        return {
            "repo": repo, "status": "failed",
            "detail": f"fetch failed after {attempts} attempts: {last_error}",
            "n_wanted": len(wanted), "n_present": len(present), "n_fetched": 0,
            "seconds": round(time.monotonic() - started, 1),
        }

    still_missing = [commit for commit in missing if not has_commit(mirror, commit)]
    status = "complete" if not still_missing else "failed"
    return {
        "repo": repo,
        "status": status,
        "detail": "" if not still_missing else f"absent after fetch: {still_missing}",
        "n_wanted": len(wanted),
        "n_present": len(present),
        "n_fetched": len(missing) - len(still_missing),
        "seconds": round(time.monotonic() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-fetch timeout in seconds.")
    args = parser.parse_args()

    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in read_jsonl(args.manifest):
        by_repo[task["repo"]].append(task)
    args.mirror_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                fetch_repo, repo, rows, args.mirror_root, args.attempts, args.timeout,
            ): repo
            for repo, rows in sorted(by_repo.items())
        }
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            results.append(row)
            print(
                f"[{row['repo']}] {row['status']} "
                f"present={row['n_present']}/{row['n_wanted']} "
                f"fetched={row['n_fetched']} {row.get('seconds', 0)}s"
                + (f" :: {row['detail']}" if row["status"] == "failed" else ""),
                flush=True,
            )

    results.sort(key=lambda row: row["repo"])
    status_path = args.mirror_root.parent / "fetch_mirrors_status.jsonl"
    status_path.write_text("".join(json.dumps(row) + "\n" for row in results))
    n_failed = sum(row["status"] != "complete" for row in results)
    print(
        f"repos={len(results)} complete={len(results) - n_failed} "
        f"failed={n_failed} status={status_path}"
    )
    return int(n_failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
