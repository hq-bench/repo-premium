#!/usr/bin/env python3
"""Verify exact-base scale-up corpora and build retrieval manifests.

Two enumeration backends produce the identical corpus file list:

* ``--mirror-root`` (preferred) reads the file list straight out of the bare
  mirror with ``git ls-tree`` at the exact base commit. No working tree is
  materialized, so the audit costs one pack traversal per task.
* ``--repo-root`` (original) walks a materialized worktree. Correct, but
  materializing 360 full worktrees on the shared filesystem is prohibitively
  slow.

``verify_lstree_equivalence.py`` checks the two agree on every task that has
both available. Either way the audit fails closed: retrieval must consume only
the emitted ``eligible_main.tsv``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

# Git tree entry modes that materialize as a readable regular file. Symlinks
# (120000) and submodule gitlinks (160000) are deliberately excluded: neither
# yields readable source text in a language-filtered checkout.
BLOB_MODES = {"100644", "100755"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_exts(raw: str) -> tuple[str, ...]:
    return tuple(
        part.strip() if part.strip().startswith(".") else f".{part.strip()}"
        for part in raw.split(",")
        if part.strip()
    )


def git(*args: str, cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        ("git",) + args, cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    return proc.returncode, proc.stdout


def list_corpus_worktree(clone_dir: Path, exts: tuple[str, ...]) -> list[str]:
    """Mirror retrieval's sorted, extension-based corpus enumeration."""
    return sorted({
        path.relative_to(clone_dir).as_posix()
        for ext in exts
        for path in clone_dir.rglob(f"*{ext}")
    })


def list_corpus_tree(mirror: Path, commit: str, exts: tuple[str, ...]) -> list[str]:
    """Same enumeration, read from the bare mirror without a checkout."""
    code, out = git("ls-tree", "-r", "-z", commit, cwd=mirror)
    if code != 0:
        raise RuntimeError(f"ls-tree failed at {commit}: {out.strip()[:400]}")
    paths: set[str] = set()
    for record in out.split("\0"):
        if not record:
            continue
        meta, _, path = record.partition("\t")
        if path and path.endswith(exts) and meta.split(" ", 1)[0] in BLOB_MODES:
            paths.add(path)
    return sorted(paths)


def mirror_path(mirror_root: Path, repo: str) -> Path:
    return mirror_root / f"{repo.replace('/', '__')}.git"


def audit_via_mirror(
    task: dict[str, Any], mirror_root: Path, exts: tuple[str, ...]
) -> tuple[list[str], str, str]:
    """Return (corpus paths, resolved commit, corpus reference)."""
    mirror = mirror_path(mirror_root, task["repo"])
    if not mirror.is_dir():
        return [], "", str(mirror)
    # Reading by SHA is a strictly stronger exactness guarantee than comparing
    # a checkout's HEAD: there is no working tree that could have drifted.
    code, kind = git("cat-file", "-t", task["base_commit"], cwd=mirror)
    if code != 0 or kind.strip() != "commit":
        return [], "", str(mirror)
    return list_corpus_tree(mirror, task["base_commit"], exts), task["base_commit"], str(mirror)


def audit_via_worktree(
    task: dict[str, Any], repo_root: Path, exts: tuple[str, ...]
) -> tuple[list[str], str, str]:
    checkout = repo_root / task["task_id"]
    if not checkout.is_dir():
        return [], "", str(checkout)
    code, head = git("rev-parse", "HEAD", cwd=checkout)
    head = head.strip() if code == 0 else ""
    if head != task["base_commit"]:
        return [], head, str(checkout)
    return list_corpus_worktree(checkout, exts), head, str(checkout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path,
                        help="Materialized worktree root (original backend).")
    parser.add_argument("--mirror-root", type=Path,
                        help="Bare mirror root (preferred, no checkout needed).")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if bool(args.repo_root) == bool(args.mirror_root):
        parser.error("pass exactly one of --repo-root or --mirror-root")
    mode = "mirror" if args.mirror_root else "worktree"

    tasks = read_jsonl(args.tasks)
    results: list[dict[str, Any]] = []
    eligible: list[tuple[dict[str, Any], str]] = []
    for task in tasks:
        exts = normalize_exts(task["code_exts"])
        error = ""
        try:
            if mode == "mirror":
                paths, head, ref = audit_via_mirror(task, args.mirror_root, exts)
            else:
                paths, head, ref = audit_via_worktree(task, args.repo_root, exts)
        except RuntimeError as exc:
            paths, head, ref, error = [], "", "", str(exc)

        path_set = set(paths)
        missing_gold = [p for p in task["main_gold_files"] if p not in path_set]
        ready = head == task["base_commit"] and bool(paths) and not missing_gold
        present_exts = sorted({
            ext for ext in exts if any(p.endswith(ext) for p in paths)
        })
        results.append({
            "effective_exts": ",".join(present_exts),
            "task_id": task["task_id"],
            "repo": task["repo"],
            "split": task["split"],
            "corpus_mode": mode,
            "corpus_ref": ref,
            "expected_base_commit": task["base_commit"],
            "actual_head": head,
            "head_matches": head == task["base_commit"],
            "code_exts": task["code_exts"],
            "n_corpus_files": len(paths),
            "n_gold_files": len(task["main_gold_files"]),
            "missing_gold_from_corpus": missing_gold,
            "error": error,
            "ready": ready,
        })
        if ready:
            eligible.append((task, ref))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "corpus_audit.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results)
    )

    # n_corpus_files travels with the manifest so the retrieval job can assert
    # that the corpus it materializes is exactly the corpus that was audited.
    #
    # effective_exts travels with it because `git archive` aborts if ANY
    # pathspec matches nothing, and a repository need not contain every
    # configured extension at every commit (axios has no .tsx). Driving the
    # pathspec from what the audit actually enumerated keeps materialization
    # consistent with the audit by construction.
    header = (
        "index\ttask_id\tsplit\tlanguage\tcode_exts\tqrel_file"
        "\tcorpus_mode\tcorpus_ref\tbase_commit\tn_corpus_files\teffective_exts"
    )
    tsv = [header]
    n_files_by_task = {row["task_id"]: row["n_corpus_files"] for row in results}
    eff_exts_by_task = {row["task_id"]: row["effective_exts"] for row in results}
    qrel_dir = args.out_dir / "qrels_paired"
    qrel_dir.mkdir(exist_ok=True)
    for index, (task, ref) in enumerate(eligible):
        qrel_path = qrel_dir / f"{task['task_id']}.jsonl"
        qrels = []
        for query_view, query in (
            ("curated", task["curated_query"]),
            ("raw", task["raw_query"]),
        ):
            qrels.append({
                "query_id": f"{task['task_id']}::{query_view}",
                "task_id": task["task_id"],
                "query_view": query_view,
                "repo": task["repo"],
                "problem_statement": query,
                "gold_files": task["main_gold_files"],
                "base_commit": task["base_commit"],
                "reference_view": "main",
                "n_gold_files": len(task["main_gold_files"]),
                "all_files_exist": True,
                "missing_files": [],
                "filter_reason": None,
                "split": task["split"],
            })
        qrel_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in qrels)
        )
        tsv.append("\t".join([
            str(index), task["task_id"], task["split"], task["language_group"],
            task["code_exts"], str(qrel_path), mode, ref, task["base_commit"],
            str(n_files_by_task[task["task_id"]]),
            eff_exts_by_task[task["task_id"]],
        ]))
    (args.out_dir / "eligible_main.tsv").write_text("\n".join(tsv) + "\n")

    summary = {
        "corpus_mode": mode,
        "n_tasks": len(tasks),
        "n_paired_queries": 2 * len(eligible),
        "n_ready": len(eligible),
        "n_failed": len(tasks) - len(eligible),
        "ready_by_split": dict(Counter(task["split"] for task, _ in eligible)),
        "n_corpus_files_total_across_snapshots": sum(r["n_corpus_files"] for r in results),
        "failed_task_ids": [r["task_id"] for r in results if not r["ready"]],
    }
    (args.out_dir / "corpus_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return int(len(eligible) != len(tasks))


if __name__ == "__main__":
    raise SystemExit(main())
