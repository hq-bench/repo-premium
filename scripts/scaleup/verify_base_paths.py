#!/usr/bin/env python3
"""Verify selected gold paths at their exact GitHub base commits."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode


def gh_json(endpoint: str, timeout: int) -> Any:
    result = subprocess.run(
        ["gh", "api", endpoint],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(message or f"gh api exited {result.returncode}")
    return json.loads(result.stdout)


def verify(row: dict[str, Any], timeout: int) -> dict[str, Any]:
    repo = row["repo"]
    commit = row["base_commit"]
    paths = row.get("main_gold_files") or []
    verified: list[str] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        endpoint = (
            f"repos/{repo}/contents/{quote(path, safe='/')}?"
            + urlencode({"ref": commit})
        )
        try:
            payload = gh_json(endpoint, timeout)
            if isinstance(payload, dict) and payload.get("type") == "file":
                verified.append(path)
            else:
                failures.append({"path": path, "error": "not a file"})
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            failures.append({"path": path, "error": str(exc)[:500]})
    checked = dict(row)
    checked["verified_main_gold_files"] = verified
    checked["base_verification_failures"] = failures
    checked["base_verified"] = bool(paths) and not failures
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    checked: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(verify, row, args.timeout): row for row in rows}
        for number, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                checked.append(future.result())
            except Exception as exc:  # Preserve unexpected per-task failures.
                failed = dict(row)
                failed["verified_main_gold_files"] = []
                failed["base_verification_failures"] = [
                    {"path": "", "error": f"unexpected: {exc}"[:500]}
                ]
                failed["base_verified"] = False
                checked.append(failed)
            if number % 25 == 0 or number == len(rows):
                print(f"verified {number}/{len(rows)}", flush=True)

    checked.sort(key=lambda row: (row["repo"], row["date_window"], row["task_id"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "base_verification.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in checked)
    )
    verified = [row for row in checked if row["base_verified"]]
    (args.out_dir / "verified_tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in verified)
    )

    by_repo_window: dict[str, Counter[str]] = defaultdict(Counter)
    for row in verified:
        by_repo_window[row["repo"]][row["date_window"]] += 1
    complete_repos = {
        repo
        for repo, counts in by_repo_window.items()
        if counts["pre"] == 5 and counts["post"] == 5
    }
    complete = [row for row in verified if row["repo"] in complete_repos]
    language_counts = Counter(row["language_group"] for row in complete)
    summary = {
        "n_input_tasks": len(rows),
        "n_verified_tasks": len(verified),
        "n_failed_tasks": len(rows) - len(verified),
        "n_verified_gold_paths": sum(
            len(row["verified_main_gold_files"]) for row in checked
        ),
        "n_complete_repositories": len(complete_repos),
        "n_complete_balanced_tasks": len(complete),
        "complete_counts_by_language": dict(sorted(language_counts.items())),
        "failed_task_ids": [row["task_id"] for row in checked if not row["base_verified"]],
        "gate_30_repositories": len(complete_repos) >= 30,
        "gate_40_per_language": bool(language_counts)
        and min(language_counts.values()) >= 40,
    }
    (args.out_dir / "base_verification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return int(summary["n_failed_tasks"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
