"""Select a deterministic repository-balanced pre/post task core."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load(path: Path, window: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        row["date_window"] = window
    return [row for row in rows if row.get("eligible_metadata")]


def stable_order(row: dict[str, Any], seed: int) -> str:
    value = f"{seed}:{row['task_id']}".encode()
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre", type=Path, nargs="+", required=True)
    parser.add_argument("--post", type=Path, nargs="+", required=True)
    parser.add_argument("--per-window-repo", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    source_rows = [row for path in args.pre for row in load(path, "pre")]
    source_rows += [row for path in args.post for row in load(path, "post")]
    for row in source_rows:
        by_key[(row["repo"], row["date_window"])].append(row)

    repositories = sorted({repo for repo, _ in by_key})
    selected: list[dict[str, Any]] = []
    insufficient: dict[str, dict[str, int]] = {}
    for repo in repositories:
        counts = {window: len(by_key[(repo, window)]) for window in ("pre", "post")}
        if min(counts.values()) < args.per_window_repo:
            insufficient[repo] = counts
            continue
        for window in ("pre", "post"):
            candidates = sorted(
                by_key[(repo, window)], key=lambda row: stable_order(row, args.seed)
            )
            selected.extend(candidates[:args.per_window_repo])

    selected.sort(key=lambda row: (row["repo"], row["date_window"], row["task_id"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "balanced_tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in selected)
    )
    language_counts = Counter(row["language_group"] for row in selected)
    summary = {
        "seed": args.seed,
        "per_window_repo": args.per_window_repo,
        "n_tasks": len(selected),
        "n_repositories": len({row["repo"] for row in selected}),
        "counts_by_window": dict(Counter(row["date_window"] for row in selected)),
        "counts_by_language": dict(sorted(language_counts.items())),
        "insufficient_repositories": insufficient,
        "gate_30_repositories": len({row["repo"] for row in selected}) >= 30,
        "gate_40_per_language": bool(language_counts) and min(language_counts.values()) >= 40,
    }
    (args.out_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return int(not (summary["gate_30_repositories"] and summary["gate_40_per_language"]))


if __name__ == "__main__":
    raise SystemExit(main())
