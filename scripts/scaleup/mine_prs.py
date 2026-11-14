"""Mine balanced PR-task candidates with base-visible file labels."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


CODE_EXTS = {
    ".c", ".cc", ".cpp", ".cxx", ".go", ".h", ".hpp", ".java", ".js",
    ".jsx", ".kt", ".mjs", ".py", ".pyi", ".rs", ".scala", ".ts", ".tsx",
}
DOC_RE = re.compile(r"^(docs?/|.*\.(md|rst|txt|adoc)$)", re.IGNORECASE)
CI_RE = re.compile(r"^(\.github/|\.gitlab/|ci/|\.ci/)", re.IGNORECASE)
TEST_RE = re.compile(r"(^|/)tests?(/|$)|(^|/)(test_|spec/)|(_test|_spec)\.", re.IGNORECASE)


def gh_json(*args: str) -> Any:
    proc = subprocess.run(
        ["gh", "api", *args], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, timeout=90,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or f"gh api exited {proc.returncode}")
    return json.loads(proc.stdout)


def repositories(path: Path) -> list[tuple[str, str]]:
    spec = json.loads(path.read_text())
    return [(repo, language) for language, repos in spec.items() for repo in repos]


def pr_numbers(repo: str, start: str, end: str | None, limit: int) -> list[int]:
    merged = f"{start}..{end}" if end else f">={start}"
    data = gh_json(
        "-X", "GET", "search/issues", "-f",
        f"q=repo:{repo} is:pr is:merged merged:{merged}",
        "-f", f"per_page={min(limit, 100)}", "-f", "sort=created",
        "-f", "order=desc",
    )
    return [int(item["number"]) for item in data.get("items", [])[:limit]]


def classify(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if DOC_RE.search(path):
        return "documentation"
    if CI_RE.search(path):
        return "ci_config"
    if TEST_RE.search(path):
        return "test"
    if suffix in CODE_EXTS:
        return "production_code"
    return "other"


def mine_one(repo: str, language: str, number: int) -> dict[str, Any]:
    pr = gh_json(f"repos/{repo}/pulls/{number}")
    files = gh_json("--paginate", f"repos/{repo}/pulls/{number}/files")
    changed: list[dict[str, Any]] = []
    main_gold: list[str] = []
    for item in files:
        status = item.get("status", "modified")
        filename = item["filename"]
        base_path = item.get("previous_filename", filename) if status == "renamed" else filename
        category = classify(base_path)
        row = {
            "path": filename,
            "base_path": base_path,
            "status": status,
            "category": category,
            "additions": int(item.get("additions", 0)),
            "deletions": int(item.get("deletions", 0)),
        }
        changed.append(row)
        if status != "added" and category == "production_code":
            main_gold.append(base_path)

    title = pr.get("title") or ""
    body = pr.get("body") or ""
    reasons: list[str] = []
    if len(body) < 100:
        reasons.append("body_too_short")
    if not 1 <= len(changed) <= 20:
        reasons.append("changed_file_count")
    if not main_gold:
        reasons.append("no_base_visible_production_code")
    return {
        "task_id": f"{repo.replace('/', '__')}-pr-{number}",
        "repo": repo,
        "repository_url": f"https://github.com/{repo}.git",
        "language_group": language,
        "pr_number": number,
        "base_commit": pr.get("base", {}).get("sha", ""),
        "merged_at": pr.get("merged_at", ""),
        "raw_query": (title + "\n\n" + body).strip(),
        "title": title,
        "body": body,
        "changed_files": changed,
        "main_gold_files": sorted(set(main_gold)),
        "reference_loc": sum(x["additions"] + x["deletions"] for x in changed),
        "filter_reasons": reasons,
        "eligible_metadata": not reasons,
    }


def mine_repo(repo: str, language: str, start: str, end: str | None,
              limit: int) -> tuple[str, list[dict[str, Any]], list[str]]:
    try:
        numbers = pr_numbers(repo, start, end, limit)
    except Exception as exc:
        return repo, [], [f"list: {exc}"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for number in numbers:
        try:
            rows.append(mine_one(repo, language, number))
        except Exception as exc:
            errors.append(f"PR {number}: {exc}")
    return repo, rows, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", type=Path, required=True)
    parser.add_argument("--merged-start", required=True)
    parser.add_argument("--merged-end")
    parser.add_argument("--max-per-repo", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = repositories(args.repos)
    rows: list[dict[str, Any]] = []
    errors: dict[str, list[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(mine_repo, repo, language, args.merged_start,
                        args.merged_end, args.max_per_repo): repo
            for repo, language in specs
        }
        for future in concurrent.futures.as_completed(futures):
            repo, repo_rows, repo_errors = future.result()
            rows.extend(repo_rows)
            if repo_errors:
                errors[repo] = repo_errors
            kept = sum(row["eligible_metadata"] for row in repo_rows)
            print(
                f"[ready] {repo}: fetched={len(repo_rows)} eligible={kept} "
                f"errors={len(repo_errors)}", flush=True
            )
            checkpoint_rows = sorted(
                rows, key=lambda row: (row["repo"], row["merged_at"], row["pr_number"])
            )
            (args.out_dir / "candidates.partial.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in checkpoint_rows)
            )

    rows.sort(key=lambda row: (row["repo"], row["merged_at"], row["pr_number"]))
    manifest = args.out_dir / "candidates.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {
        "merged_start": args.merged_start,
        "merged_end": args.merged_end,
        "n_repositories_requested": len(specs),
        "n_repositories_returned": len({row["repo"] for row in rows}),
        "n_candidates": len(rows),
        "n_eligible_metadata": sum(row["eligible_metadata"] for row in rows),
        "eligible_by_language": dict(Counter(
            row["language_group"] for row in rows if row["eligible_metadata"]
        )),
        "errors": errors,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    completely_failed = [repo for repo, _ in specs if not any(r["repo"] == repo for r in rows)]
    return int(bool(completely_failed))


if __name__ == "__main__":
    raise SystemExit(main())
