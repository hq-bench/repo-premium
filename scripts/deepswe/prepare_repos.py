"""Clone DeepSWE repositories and pin one checkout per task base commit."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path
from typing import Any


def run(*args: str, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        args, cwd=cwd, check=False, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode:
        output = proc.stdout.strip()
        detail = output[-4000:] if output else "no command output"
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(args)}\n{detail}"
        )
    return proc.stdout.strip()


def prepare(row: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    task_id = row["task_id"]
    target = repo_root / task_id
    result = {"task_id": task_id, "target": str(target), "status": "failed", "detail": ""}
    try:
        created = False
        if target.exists():
            if not (target / ".git").exists():
                raise RuntimeError("target exists but is not a git checkout")
            dirty = run("git", "status", "--porcelain", cwd=target)
            if dirty:
                raise RuntimeError("existing checkout is dirty; refusing to modify it")
        else:
            run("git", "clone", "--filter=blob:none", "--no-checkout",
                row["repository_url"], str(target))
            created = True
        head = run("git", "rev-parse", "HEAD", cwd=target) if (target / ".git" / "HEAD").exists() else ""
        if created or head != row["base_commit"]:
            run("git", "fetch", "--depth", "1", "origin", row["base_commit"], cwd=target)
            run("git", "checkout", "--detach", row["base_commit"], cwd=target)
        actual = run("git", "rev-parse", "HEAD", cwd=target)
        if actual != row["base_commit"]:
            raise RuntimeError(f"checkout mismatch: expected {row['base_commit']}, got {actual}")
        result.update(status="ready", detail=actual)
    except Exception as exc:  # retain all failures in the resumable status report
        result["detail"] = str(exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task-id", action="append", default=[])
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.task_id:
        wanted = set(args.task_id)
        rows = [row for row in rows if row["task_id"] in wanted]
    if args.limit > 0:
        rows = rows[:args.limit]
    args.repo_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare, row, args.repo_root) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{result['status']}] {result['task_id']}: {result['detail']}")
    results.sort(key=lambda row: row["task_id"])
    status_path = args.repo_root / "prepare_status.jsonl"
    status_path.write_text("".join(json.dumps(row) + "\n" for row in results))
    failed = sum(row["status"] != "ready" for row in results)
    print(f"ready={len(results) - failed} failed={failed} status={status_path}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
