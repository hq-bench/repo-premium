"""Audit DeepSWE tasks and build file-level reference-edit qrels.

The official task tree supplies one ``solution/solution.patch`` per task. This
script extracts its changed files, separates searchable base-snapshot files from
new files, classifies file roles, and emits qrel variants plus cluster manifests.

Run once without ``--repo-root`` to prepare the clone manifest, then rerun after
``prepare_repos.py`` to verify every gold path against its pinned base checkout.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10 cluster fallback
    import tomli as tomllib  # type: ignore[no-redef]


SOURCE_EXTS = {
    ".c", ".cc", ".cpp", ".cs", ".go", ".h", ".hpp", ".java", ".js",
    ".jsx", ".mjs", ".cjs", ".py", ".pyi", ".rs", ".ts", ".tsx",
}
DOC_EXTS = {".md", ".mdx", ".rst", ".txt", ".adoc"}
CONFIG_EXTS = {
    ".json", ".json5", ".lock", ".toml", ".yaml", ".yml", ".ini", ".cfg",
}
LANGUAGE_EXTS = {
    "python": ".py,.pyi",
    "go": ".go",
    "rust": ".rs",
    "typescript": ".ts,.tsx,.js,.jsx,.mjs,.cjs",
    "javascript": ".js,.jsx,.mjs,.cjs,.ts,.tsx",
}
ADMIN_FOOTER = (
    "IMPORTANT: Please work on this in a new branch from main and commit "
    "everything when you are done."
)
TEST_PARTS = {"test", "tests", "testing", "__tests__", "spec", "specs"}
DOC_PARTS = {"doc", "docs", "documentation"}
CONFIG_NAMES = {
    "dockerfile", "makefile", "justfile", "package.json", "pyproject.toml",
    "cargo.toml", "go.mod", "go.sum", "tsconfig.json",
}


@dataclass
class ChangedFile:
    path: str
    old_path: str
    status: str
    category: str
    additions: int
    deletions: int


def _diff_path(token: str) -> str:
    if token in {"/dev/null", "dev/null"}:
        return "/dev/null"
    return token[2:] if token.startswith(("a/", "b/")) else token


def classify_path(path: str) -> str:
    p = PurePosixPath(path)
    parts = {part.lower() for part in p.parts}
    name = p.name.lower()
    suffix = p.suffix.lower()
    test_named = (
        bool(parts & TEST_PARTS)
        or name.startswith("test_")
        or name.endswith("_test.go")
        or bool(re.search(r"\.(test|spec)\.[^.]+$", name))
    )
    if test_named:
        return "test"
    if parts & DOC_PARTS or suffix in DOC_EXTS:
        return "documentation"
    if name in CONFIG_NAMES or suffix in CONFIG_EXTS or ".github" in parts:
        return "config_build"
    if suffix in SOURCE_EXTS:
        return "production_code"
    return "other"


def parse_patch(patch: str) -> list[ChangedFile]:
    headers = list(re.finditer(r"^diff --git (.+)$", patch, flags=re.MULTILINE))
    changed: list[ChangedFile] = []
    for i, match in enumerate(headers):
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(patch)
        block = patch[match.start():block_end]
        try:
            tokens = shlex.split(match.group(1))
        except ValueError:
            tokens = match.group(1).split()
        if len(tokens) < 2:
            continue
        old_path, new_path = _diff_path(tokens[0]), _diff_path(tokens[1])
        if re.search(r"^new file mode ", block, flags=re.MULTILINE) or "--- /dev/null" in block:
            status = "added"
        elif re.search(r"^deleted file mode ", block, flags=re.MULTILINE) or "+++ /dev/null" in block:
            status = "deleted"
        elif re.search(r"^rename from ", block, flags=re.MULTILINE):
            status = "renamed"
        else:
            status = "modified"
        path = old_path if status == "deleted" else new_path
        additions = deletions = 0
        for line in block.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                continue
            additions += int(line.startswith("+"))
            deletions += int(line.startswith("-"))
        changed.append(ChangedFile(
            path=path,
            old_path=old_path,
            status=status,
            category=classify_path(path),
            additions=additions,
            deletions=deletions,
        ))
    return changed


def repo_slug(repository_url: str) -> str:
    clean = repository_url.rstrip("/")
    if clean.endswith(".git"):
        clean = clean[:-4]
    bits = clean.split("/")
    return "/".join(bits[-2:])


def qrel_record(
    task: dict[str, Any], gold_files: list[str], missing_files: list[str], view: str,
    verified: bool,
) -> dict[str, Any]:
    return {
        "query_id": task["task_id"],
        "repo": task["repo"],
        "problem_statement": task["instruction"],
        "raw_problem_statement": task["raw_instruction"],
        "gold_files": gold_files,
        "base_commit": task["base_commit"],
        "language": task["language"],
        "reference_view": view,
        "n_gold_files": len(gold_files),
        "all_files_exist": bool(gold_files) and verified and not missing_files,
        "missing_files": missing_files,
        "filter_reason": None if gold_files else "no_gold_files_for_view",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def median(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def code_extensions(paths: list[str], language: str) -> str:
    """Prefer observed gold-path extensions over occasionally stale metadata."""
    observed = sorted({PurePosixPath(path).suffix.lower() for path in paths
                       if PurePosixPath(path).suffix.lower() in SOURCE_EXTS})
    return ",".join(observed) or LANGUAGE_EXTS.get(
        language, ".py,.go,.rs,.ts,.tsx,.js,.jsx"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path,
                        help="Optional <task-id>/ checkout root for filesystem verification")
    args = parser.parse_args()

    task_dirs = sorted(p for p in args.tasks_root.iterdir() if (p / "task.toml").is_file())
    if not task_dirs:
        raise SystemExit(f"no DeepSWE tasks found under {args.tasks_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    qrels_by_view: dict[str, list[dict[str, Any]]] = {
        "main": [], "all_existing": [], "full_reference": [],
    }
    categories: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    languages: Counter[str] = Counter()

    for task_dir in task_dirs:
        meta = tomllib.loads((task_dir / "task.toml").read_text())
        task_meta = meta.get("metadata", {})
        task_id = str(task_meta.get("task_id") or task_dir.name)
        repository_url = str(task_meta["repository_url"])
        base_commit = str(task_meta["base_commit_hash"])
        language = str(task_meta.get("language", "unknown")).lower()
        raw_instruction = (task_dir / "instruction.md").read_text().strip()
        instruction = raw_instruction
        if instruction.endswith(ADMIN_FOOTER):
            instruction = instruction[:-len(ADMIN_FOOTER)].rstrip()
        patch_path = task_dir / "solution" / "solution.patch"
        patch = patch_path.read_text(errors="replace")
        changed = parse_patch(patch)
        for item in changed:
            categories[item.category] += 1
            statuses[item.status] += 1
        languages[language] += 1

        main_files = [
            f.path for f in changed
            if f.category == "production_code" and f.status != "added"
        ]
        all_existing = [f.path for f in changed if f.status != "added"]
        full_reference = [f.path for f in changed]
        checkout = args.repo_root / task_id if args.repo_root else None
        verified = bool(checkout and checkout.is_dir())

        def missing(files: list[str]) -> list[str]:
            if not verified or checkout is None:
                return list(files)
            return [path for path in files if not (checkout / path).exists()]

        task = {
            "task_id": task_id,
            "display_title": str(task_meta.get("display_title", "")),
            "repo": repo_slug(repository_url),
            "repository_url": repository_url,
            "base_commit": base_commit,
            "language": language,
            "instruction": instruction,
            "raw_instruction": raw_instruction,
            "solution_patch": str(patch_path),
        }
        qrel_views = {
            "main": main_files,
            "all_existing": all_existing,
            "full_reference": full_reference,
        }
        view_records: dict[str, dict[str, Any]] = {}
        for view, files in qrel_views.items():
            miss = missing(files)
            rec = qrel_record(task, files, miss, view, verified)
            view_records[view] = rec
            qrels_by_view[view].append(rec)
            write_jsonl(args.out_dir / f"qrels_{view}" / f"{task_id}.jsonl", [rec])

        record = {
            **{k: v for k, v in task.items() if k not in {"instruction", "raw_instruction"}},
            "instruction_chars": len(instruction),
            "raw_instruction_chars": len(raw_instruction),
            "admin_footer_stripped": raw_instruction != instruction,
            "changed_files": [asdict(item) for item in changed],
            "n_changed_files": len(changed),
            "reference_loc": sum(f.additions + f.deletions for f in changed),
            "main_gold_files": main_files,
            "all_existing_gold_files": all_existing,
            "new_files": [f.path for f in changed if f.status == "added"],
            "checkout_verified": verified,
            "main_missing_files": view_records["main"]["missing_files"],
            "main_eligible_patch": bool(main_files),
            "main_eligible_verified": view_records["main"]["all_files_exist"],
        }
        records.append(record)
        manifests.append({
            "task_id": task_id,
            "repo": task["repo"],
            "repository_url": repository_url,
            "base_commit": base_commit,
            "language": language,
            "code_exts": code_extensions(main_files, language),
        })

    write_jsonl(args.out_dir / "tasks_audit.jsonl", records)
    write_jsonl(args.out_dir / "task_manifest.jsonl", manifests)
    for view, rows in qrels_by_view.items():
        write_jsonl(args.out_dir / f"qrels_{view}.jsonl", rows)

    eligible = [r for r in records if r["main_eligible_verified"]]
    tsv = ["index\ttask_id\tlanguage\tcode_exts\tqrel_file\trepo_dir"]
    for idx, row in enumerate(eligible):
        task_id = row["task_id"]
        exts = code_extensions(row["main_gold_files"], row["language"])
        tsv.append("\t".join([
            str(idx), task_id, row["language"], exts,
            str(args.out_dir / "qrels_main" / f"{task_id}.jsonl"),
            str(args.repo_root / task_id),
        ]))
    (args.out_dir / "eligible_main.tsv").write_text("\n".join(tsv) + "\n")

    summary = {
        "source_tasks_root": str(args.tasks_root),
        "repo_root": str(args.repo_root) if args.repo_root else None,
        "n_tasks": len(records),
        "n_repositories": len({r["repo"] for r in records}),
        "languages": dict(sorted(languages.items())),
        "changed_file_categories": dict(sorted(categories.items())),
        "changed_file_statuses": dict(sorted(statuses.items())),
        "n_main_patch_eligible": sum(r["main_eligible_patch"] for r in records),
        "n_main_verified_eligible": len(eligible),
        "n_with_new_files": sum(bool(r["new_files"]) for r in records),
        "n_admin_footers_stripped": sum(r["admin_footer_stripped"] for r in records),
        "median_changed_files": median([r["n_changed_files"] for r in records]),
        "median_reference_loc": median([r["reference_loc"] for r in records]),
        "headline_gate_at_least_70": len(eligible) >= 70 if args.repo_root else None,
    }
    (args.out_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    report = [
        "# DeepSWE Phase-0 audit", "",
        f"- Tasks: **{summary['n_tasks']}**",
        f"- Repositories: **{summary['n_repositories']}**",
        f"- Languages: `{summary['languages']}`",
        f"- Patch-eligible main qrels: **{summary['n_main_patch_eligible']}**",
        f"- Filesystem-verified main qrels: **{summary['n_main_verified_eligible']}**",
        f"- Tasks containing new reference files: **{summary['n_with_new_files']}**",
        f"- Median changed files: **{summary['median_changed_files']}**",
        f"- Median reference LOC: **{summary['median_reference_loc']}**",
        "", "Main qrels contain only modified/deleted production-code files.",
        "Reference patches are one valid implementation, not exhaustive relevance.", "",
    ]
    (args.out_dir / "AUDIT_REPORT.md").write_text("\n".join(report))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
