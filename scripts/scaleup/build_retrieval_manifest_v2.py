#!/usr/bin/env python3
"""Build retrieval manifest v2: repo-level extension union + task replacements.

The v1 manifest assigned one language's extensions per repository, so 30 tasks
in 11 polyglot repositories had gold files the retrieval corpus could never
contain. Two distinct fixes apply, because the two situations are not the same
problem:

* **Extension union** (10 repositories). The uncovered gold is genuine
  first-party source in a second language -- Scala in Flink and Kafka, Kotlin
  in Gradle, front-end JavaScript in Gitea and Quarkus, C/C++ extension code in
  Pillow, pandas and SciPy. The corpus definition was too narrow, so each
  affected repository's ``code_exts`` is widened to cover every gold extension
  it uses. The union is applied at repository level, not task level: two tasks
  in one repository must search the same file set or the repository-blocked
  date comparison is no longer a controlled comparison.

* **Gold restriction plus replacement** (ruff). The uncovered gold is parser
  test fixtures under ``resources/``, which the design excludes from main qrels
  by rule. Those paths are dropped rather than admitted to the corpus, and any
  task left with gold identical to an earlier task in its cell is swapped for a
  deterministic replacement (see ``select_replacements.py``).

v1 is never mutated. The output is a new manifest with its own checksum, plus a
provenance record naming every change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def exts_of(code_exts: str) -> set[str]:
    return {
        part.strip() if part.strip().startswith(".") else f".{part.strip()}"
        for part in code_exts.split(",") if part.strip()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--plan", type=Path,
                        help="Replacement plan from select_replacements.py.")
    parser.add_argument("--replacement-candidates", type=Path,
                        help="Candidate records for the replacement tasks.")
    parser.add_argument("--replacement-labels", type=Path,
                        help="Curated queries for the replacement tasks.")
    parser.add_argument("--restrict-gold-repo", action="append", default=[],
                        help="Repositories where uncovered gold is dropped "
                             "instead of widening the corpus (repeatable).")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    tasks = read_jsonl(args.tasks)
    restrict = set(args.restrict_gold_repo)

    plan = read_jsonl(args.plan) if args.plan else []
    replaced = {row["replace_task_id"]: row for row in plan if row["status"] == "resolved"}
    candidates = {
        row["task_id"]: row
        for row in (read_jsonl(args.replacement_candidates)
                    if args.replacement_candidates else [])
    }
    labels = {
        row["task_id"]: row
        for row in (read_jsonl(args.replacement_labels)
                    if args.replacement_labels else [])
    }

    # ---- Pass 1: repository-level target extension set -------------------
    repo_exts: dict[str, set[str]] = defaultdict(set)
    repo_gold_exts: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        repo_exts[task["repo"]] |= exts_of(task["code_exts"])
        for gold in task["main_gold_files"]:
            repo_gold_exts[task["repo"]].add(os.path.splitext(gold)[1] or "(none)")
    # Replacement gold participates in the union too.
    for old, row in replaced.items():
        repo = row["repo"]
        for gold in row["replacement_gold"] or []:
            repo_gold_exts[repo].add(os.path.splitext(gold)[1] or "(none)")

    widened: dict[str, dict[str, Any]] = {}
    for repo, current in repo_exts.items():
        if repo in restrict:
            continue
        added = {e for e in repo_gold_exts[repo] - current if e != "(none)"}
        if added:
            widened[repo] = {"from": sorted(current), "added": sorted(added)}
            repo_exts[repo] = current | added

    # ---- Pass 2: emit v2 rows -------------------------------------------
    out_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    missing_labels: list[str] = []

    for task in tasks:
        repo = task["repo"]
        if task["task_id"] in replaced:
            row = replaced[task["task_id"]]
            new_id = row["replacement_task_id"]
            cand = candidates.get(new_id)
            label = labels.get(new_id)
            if cand is None:
                raise SystemExit(f"missing candidate record for replacement {new_id}")
            if label is None:
                missing_labels.append(new_id)
                continue
            gold = [g for g in cand["main_gold_files"]
                    if g.endswith(tuple(repo_exts[repo]))]
            new_task = dict(task)
            new_task.update({
                "task_id": new_id,
                "pr_number": cand["pr_number"],
                "base_commit": cand["base_commit"],
                "raw_query": cand["raw_query"],
                "curated_query": label["curated_query"],
                "main_gold_files": sorted(gold),
                "code_exts": ",".join(sorted(repo_exts[repo])),
                "replacement_of": task["task_id"],
                "replacement_reason": row["reason"],
            })
            out_rows.append(new_task)
            changes.append({
                "kind": "replacement", "repo": repo,
                "from": task["task_id"], "to": new_id, "reason": row["reason"],
            })
            continue

        new_task = dict(task)
        # Rewrite only when the extension *set* changes. Re-serializing an
        # unchanged set in sorted order would add 60 no-op diffs to the
        # provenance record and obscure the real ones.
        if repo_exts[repo] != exts_of(task["code_exts"]):
            new_exts = ",".join(sorted(repo_exts[repo]))
            new_task["code_exts"] = new_exts
            changes.append({
                "kind": "code_exts", "repo": repo, "task_id": task["task_id"],
                "from": task["code_exts"], "to": new_exts,
            })
        if repo in restrict:
            kept = [g for g in task["main_gold_files"]
                    if g.endswith(tuple(repo_exts[repo]))]
            if len(kept) != len(task["main_gold_files"]):
                dropped = sorted(set(task["main_gold_files"]) - set(kept))
                new_task["main_gold_files"] = sorted(kept)
                new_task["dropped_non_production_gold"] = dropped
                changes.append({
                    "kind": "gold_restriction", "repo": repo,
                    "task_id": task["task_id"], "dropped": dropped,
                })
        out_rows.append(new_task)

    if missing_labels:
        raise SystemExit(
            "missing curated queries for replacements: "
            + ", ".join(sorted(missing_labels))
            + "\nRun curate_and_verify.py on the replacement candidates first."
        )

    # ---- Validate: every gold path is inside its task's corpus -----------
    uncovered = [
        {"task_id": t["task_id"], "repo": t["repo"],
         "gold": [g for g in t["main_gold_files"]
                  if not g.endswith(tuple(exts_of(t["code_exts"])))]}
        for t in out_rows
        if any(not g.endswith(tuple(exts_of(t["code_exts"])))
               for g in t["main_gold_files"])
    ]
    empty_gold = [t["task_id"] for t in out_rows if not t["main_gold_files"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "retrieval_tasks_v2.jsonl"
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows)
    out_path.write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()

    provenance = {
        "source_manifest": str(args.tasks),
        "source_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "output_manifest": str(out_path),
        "output_sha256": digest,
        "n_tasks": len(out_rows),
        "n_repositories": len({r["repo"] for r in out_rows}),
        "extension_union": widened,
        "gold_restricted_repositories": sorted(restrict),
        "n_changes": len(changes),
        "changes": changes,
        "validation": {
            "tasks_with_gold_outside_corpus": uncovered,
            "tasks_with_empty_gold": empty_gold,
            "passed": not uncovered and not empty_gold and len(out_rows) == len(tasks),
        },
    }
    (args.out_dir / "manifest_v2_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps({
        k: provenance[k] for k in
        ("n_tasks", "n_repositories", "extension_union",
         "gold_restricted_repositories", "n_changes", "validation")
    }, indent=2))
    print(f"\noutput_sha256 {digest}")
    return int(not provenance["validation"]["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
