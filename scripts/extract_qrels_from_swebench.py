"""
Extract file-granularity qrels from SWE-bench Verified for CoREB-Repo.

Reads `princeton-nlp/SWE-bench_Verified` and produces one JSONL per repo at
`<out_dir>/<org__repo>.jsonl`. Each record represents one query
(= one SWE-bench instance):

  {
    "query_id":         "django__django-17087",
    "repo":             "django/django",
    "problem_statement": "<issue text>",
    "gold_files":       ["django/db/models/fields/__init__.py", ...],
    "base_commit":      "4a72da71001f...",
    "created_at":       "2024-03-15T12:34:56Z",
    "difficulty":       "15 min - 1 hour",
    "patch_loc":        23,
    "n_gold_files":     1,
    "all_files_exist":  true   # all gold_files resolved against the cloned base_commit
  }

Records with `all_files_exist=false` are still written but flagged for downstream
filtering. A summary JSON is written alongside.

Pre/post-cutoff split for the contamination audit (Cell A vs Cell B in
docs/ROADMAP §3.5) uses the `created_at` field.

Usage:
  conda activate llm
  python scripts/extract_qrels_from_swebench.py \\
      --clone-root ${REPO_PREMIUM_DATA:-data}/repos \\
      --out-dir ${REPO_PREMIUM_DATA:-data}/release/qrels
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"

DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def patched_files(patch: str) -> list[str]:
    if not patch:
        return []
    paths: list[str] = []
    seen: set[str] = set()
    for a_path, b_path in DIFF_FILE_RE.findall(patch):
        p = b_path if b_path != "/dev/null" else a_path
        if p not in seen:
            seen.add(p)
            paths.append(p)
    return paths


def patch_loc(patch: str) -> int:
    if not patch:
        return 0
    n = 0
    for line in patch.splitlines():
        if not line or line.startswith("+++") or line.startswith("---"):
            continue
        if line[0] in "+-":
            n += 1
    return n


@dataclass
class QrelRecord:
    query_id: str
    repo: str
    problem_statement: str
    gold_files: list[str]
    base_commit: str
    created_at: str
    difficulty: str
    patch_loc: int
    n_gold_files: int
    all_files_exist: bool
    missing_files: list[str]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clone-root", type=Path, required=True,
                   help="Root containing <org>__<repo>/ clones (from clone_and_checkout.py)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output dir for per-repo qrel JSONLs")
    p.add_argument("--cutoff", default="2026-01-01",
                   help="ISO date for pre/post cutoff split in summary")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {DATASET_NAME} split={SPLIT}", file=sys.stderr)
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    rows = list(ds)
    print(f"[loaded] {len(rows)} instances", file=sys.stderr)

    by_repo: dict[str, list[QrelRecord]] = collections.defaultdict(list)
    skipped_empty_patch = 0

    for row in rows:
        files = patched_files(row.get("patch", ""))
        if not files:
            skipped_empty_patch += 1
            continue

        repo = row["repo"]
        clone_dir = args.clone_root / repo.replace("/", "__")
        missing = [f for f in files if not (clone_dir / f).exists()]

        rec = QrelRecord(
            query_id=row["instance_id"],
            repo=repo,
            problem_statement=row.get("problem_statement", "") or "",
            gold_files=files,
            base_commit=row.get("base_commit", "") or "",
            created_at=row.get("created_at", "") or "",
            difficulty=row.get("difficulty", "") or "",
            patch_loc=patch_loc(row.get("patch", "")),
            n_gold_files=len(files),
            all_files_exist=(len(missing) == 0),
            missing_files=missing,
        )
        by_repo[repo].append(rec)

    # Write per-repo JSONL + summary
    summary: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "total_instances_in_dataset": len(rows),
        "skipped_empty_patch": skipped_empty_patch,
        "cutoff": args.cutoff,
        "per_repo": {},
    }

    print(f"\n{'repo':<32s}  {'n':>5s}  {'all_ok':>7s}  {'pre_cut':>7s}  {'post_cut':>8s}",
          file=sys.stderr)
    print("-" * 70, file=sys.stderr)

    total_records = 0
    total_all_ok = 0
    total_pre = 0
    total_post = 0

    for repo in sorted(by_repo.keys(), key=lambda r: -len(by_repo[r])):
        records = by_repo[repo]
        out_path = args.out_dir / f"{repo.replace('/', '__')}.jsonl"
        with out_path.open("w") as f:
            for rec in records:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

        n = len(records)
        all_ok = sum(1 for r in records if r.all_files_exist)
        pre_cut = sum(1 for r in records if r.created_at and r.created_at < args.cutoff)
        post_cut = sum(1 for r in records if r.created_at and r.created_at >= args.cutoff)

        summary["per_repo"][repo] = {
            "n_qrels": n,
            "n_all_files_exist": all_ok,
            "n_pre_cutoff": pre_cut,
            "n_post_cutoff": post_cut,
            "median_gold_files": sorted(r.n_gold_files for r in records)[n // 2] if n else 0,
            "median_patch_loc": sorted(r.patch_loc for r in records)[n // 2] if n else 0,
            "out_path": str(out_path),
        }

        total_records += n
        total_all_ok += all_ok
        total_pre += pre_cut
        total_post += post_cut

        print(f"{repo:<32s}  {n:>5d}  {all_ok:>7d}  {pre_cut:>7d}  {post_cut:>8d}",
              file=sys.stderr)

    print("-" * 70, file=sys.stderr)
    print(f"{'TOTAL':<32s}  {total_records:>5d}  {total_all_ok:>7d}  "
          f"{total_pre:>7d}  {total_post:>8d}", file=sys.stderr)

    summary["totals"] = {
        "n_qrels": total_records,
        "n_all_files_exist": total_all_ok,
        "n_pre_cutoff": total_pre,
        "n_post_cutoff": total_post,
        "skipped_empty_patch": skipped_empty_patch,
    }
    summary_path = args.out_dir / "qrels_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[written] {summary_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
