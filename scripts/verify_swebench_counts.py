"""
Verify SWE-bench Verified per-repo instance counts.

Replaces the ⚠-marked estimates in docs/coreb_repo_repo_survey.md §2 with
ground-truth counts pulled directly from the official HuggingFace dataset.

Also reports per-repo:
  - difficulty distribution
  - patch size statistics (mean #files touched, mean patch LOC)
  - issue length statistics (problem_statement char count)
  - file-extension distribution of patched files (sanity check that all are Python)

Usage:
  conda activate llm
  python scripts/verify_swebench_counts.py \\
      [--out scripts/swebench_verified_stats.json] \\
      [--markdown]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"

# Regex to extract file paths from a unified diff: lines like
#   diff --git a/path/to/file.py b/path/to/file.py
DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def patched_files(patch: str) -> list[str]:
    """Return the list of file paths referenced by `diff --git` lines in a patch."""
    if not patch:
        return []
    paths: list[str] = []
    seen: set[str] = set()
    for a_path, b_path in DIFF_FILE_RE.findall(patch):
        # Prefer the post-edit path; both are usually identical
        p = b_path if b_path != "/dev/null" else a_path
        if p not in seen:
            seen.add(p)
            paths.append(p)
    return paths


def patch_loc(patch: str) -> int:
    """Approximate patch size: count lines starting with + or - (excluding diff headers)."""
    if not patch:
        return 0
    n = 0
    for line in patch.splitlines():
        if not line:
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line[0] in "+-":
            n += 1
    return n


def safe_mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def safe_median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_repo: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r)

    per_repo: dict[str, dict[str, Any]] = {}
    for repo, items in sorted(by_repo.items(), key=lambda kv: -len(kv[1])):
        files_per_patch = [len(patched_files(it["patch"])) for it in items]
        loc_per_patch = [patch_loc(it["patch"]) for it in items]
        issue_chars = [len(it.get("problem_statement") or "") for it in items]

        ext_counter: collections.Counter[str] = collections.Counter()
        for it in items:
            for p in patched_files(it["patch"]):
                ext = Path(p).suffix.lower() or "<noext>"
                ext_counter[ext] += 1

        difficulty_counter: collections.Counter[str] = collections.Counter()
        for it in items:
            d = it.get("difficulty") or "<unknown>"
            difficulty_counter[d] += 1

        per_repo[repo] = {
            "n_instances": len(items),
            "files_per_patch_mean": round(safe_mean(files_per_patch), 2),
            "files_per_patch_median": round(safe_median(files_per_patch), 2),
            "files_per_patch_max": max(files_per_patch) if files_per_patch else 0,
            "patch_loc_mean": round(safe_mean(loc_per_patch), 1),
            "patch_loc_median": round(safe_median(loc_per_patch), 1),
            "issue_chars_mean": round(safe_mean(issue_chars), 1),
            "issue_chars_median": round(safe_median(issue_chars), 1),
            "patched_file_exts": dict(ext_counter.most_common()),
            "difficulty": dict(difficulty_counter.most_common()),
        }

    return {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "total_instances": len(rows),
        "n_repos": len(per_repo),
        "per_repo": per_repo,
    }


def render_markdown(stats: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"## SWE-bench Verified — verified counts ({stats['dataset']}, "
                 f"split={stats['split']})\n")
    lines.append(f"- Total instances: **{stats['total_instances']}**")
    lines.append(f"- Unique repos: **{stats['n_repos']}**\n")

    lines.append("| Repo | # instances | files/patch (med) | patch LOC (med) | "
                 "issue chars (med) | qrel-ready? |")
    lines.append("|------|------------:|------------------:|----------------:|"
                 "------------------:|-------------|")
    for repo, m in stats["per_repo"].items():
        # qrel-ready: at least 1 file touched per patch in median + non-empty issue
        ready = "✅" if m["files_per_patch_median"] >= 1 and m["issue_chars_median"] > 50 else "⚠"
        lines.append(
            f"| `{repo}` | {m['n_instances']} | {m['files_per_patch_median']} | "
            f"{m['patch_loc_median']} | {int(m['issue_chars_median'])} | {ready} |"
        )

    lines.append("")
    lines.append("### Per-repo file extension distribution (patched files)")
    lines.append("")
    for repo, m in stats["per_repo"].items():
        exts = ", ".join(f"`{k}`×{v}" for k, v in m["patched_file_exts"].items())
        lines.append(f"- **{repo}**: {exts}")

    lines.append("")
    lines.append("### Difficulty distribution")
    lines.append("")
    for repo, m in stats["per_repo"].items():
        diff = ", ".join(f"{k}: {v}" for k, v in m["difficulty"].items())
        lines.append(f"- **{repo}**: {diff}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("scripts/swebench_verified_stats.json"),
        help="JSON output path",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also write a Markdown report next to the JSON",
    )
    args = parser.parse_args()

    print(f"[load] {DATASET_NAME} split={SPLIT}", file=sys.stderr)
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    rows: list[dict[str, Any]] = list(ds)
    print(f"[loaded] {len(rows)} rows", file=sys.stderr)

    stats = summarize(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"[written] {args.out}", file=sys.stderr)

    if args.markdown:
        md_path = args.out.with_suffix(".md")
        md_path.write_text(render_markdown(stats))
        print(f"[written] {md_path}", file=sys.stderr)

    # Stdout: compact summary table for quick copy/paste into the survey doc
    print()
    print(f"Total: {stats['total_instances']} instances across {stats['n_repos']} repos")
    print()
    for repo, m in stats["per_repo"].items():
        print(f"  {repo:40s}  n={m['n_instances']:4d}  "
              f"files/patch(med)={m['files_per_patch_median']:.1f}  "
              f"patch_loc(med)={m['patch_loc_median']:.0f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
