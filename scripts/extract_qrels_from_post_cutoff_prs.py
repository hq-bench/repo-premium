"""
Phase 1.2 — Mine post-cutoff merged PRs into file-granularity qrels.

For each repo, fetch the N most recent merged PRs with `merged_at >= cutoff`,
extract (PR body, touched files) → qrel record. Applies quality filters
(docs-only / CI-only / body too short / too many files) and verifies that
gold files exist in the pinned snapshot.

Output: `<out_dir>/<org__repo>.jsonl` + `post_cutoff_qrels_summary.json`,
schema-compatible with extract_qrels_from_swebench.py.

Usage (Cell B — post-cutoff, open-ended window):
  gh auth status
  conda run -n llm python scripts/extract_qrels_from_post_cutoff_prs.py \\
      --clone-root ${REPO_PREMIUM_DATA:-data}/repos \\
      --out-dir ${REPO_PREMIUM_DATA:-data}/release/qrels_post_cutoff \\
      --merged-start 2026-01-01 \\
      --max-per-repo 100 \\
      --workers 5

Usage (R5 — pre-cutoff negative control, closed window + repo subset):
  conda run -n llm python scripts/extract_qrels_from_post_cutoff_prs.py \\
      --clone-root ${REPO_PREMIUM_DATA:-data}/repos \\
      --out-dir ${REPO_PREMIUM_DATA:-data}/coreb_repo_qrels_r5 \\
      --merged-start 2023-07-01 \\
      --merged-end   2023-12-31 \\
      --repos django/django,sympy/sympy,sphinx-doc/sphinx,matplotlib/matplotlib,scikit-learn/scikit-learn,astropy/astropy,pydata/xarray,pytest-dev/pytest,pylint-dev/pylint,psf/requests \\
      --max-per-repo 20 \\
      --workers 5
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPOS = [
    "django/django", "sympy/sympy", "sphinx-doc/sphinx", "matplotlib/matplotlib",
    "scikit-learn/scikit-learn", "astropy/astropy", "pydata/xarray",
    "pytest-dev/pytest", "pylint-dev/pylint", "psf/requests",
    "react/react", "axios/axios", "prettier/prettier",
    "gin-gonic/gin", "spring-projects/spring-boot", "kubernetes/kubernetes",
]

CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs",
    ".go", ".java", ".kt", ".scala",
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".rs",
    ".rb", ".php", ".cs", ".swift",
}
DOC_RE = re.compile(r"^(docs/|.*\.(md|rst|txt|adoc)$)", re.IGNORECASE)
CI_RE = re.compile(r"^(\.github/|\.gitlab/|ci/|\.ci/|.*\.(yml|yaml)$)", re.IGNORECASE)
TEST_RE = re.compile(r"(^|/)tests?(/|$)|(test_|_test\.|_spec\.)", re.IGNORECASE)


@dataclass
class PRQrel:
    query_id: str
    repo: str
    problem_statement: str
    gold_files: list[str]
    base_commit: str
    created_at: str
    merged_at: str
    difficulty: str
    patch_loc: int
    n_gold_files: int
    all_files_exist: bool
    missing_files: list[str]
    filter_reason: str | None  # null if kept; else why we'd drop in downstream filter


def gh_api(args: list[str]) -> Any:
    cmd = ["gh", "api", *args]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"gh api failed: {' '.join(cmd)}\n{r.stderr.strip()}")
    return json.loads(r.stdout)


def list_pr_numbers(repo: str, merged_start: str, merged_end: str | None,
                    limit: int) -> list[int]:
    """Return up to `limit` most-recently-merged PR numbers with
    merged_at >= merged_start (and <= merged_end if provided)."""
    nums: list[int] = []
    page = 1
    if merged_end:
        merged_q = f"merged:{merged_start}..{merged_end}"
    else:
        merged_q = f"merged:>={merged_start}"
    while len(nums) < limit:
        per_page = min(100, limit - len(nums))
        data = gh_api([
            "-X", "GET", "search/issues",
            "-f", f"q=repo:{repo} is:pr is:merged {merged_q}",
            "-f", f"per_page={per_page}",
            "-f", f"page={page}",
            "-f", "sort=created",
            "-f", "order=desc",
        ])
        items = data.get("items", [])
        if not items:
            break
        nums.extend(int(it["number"]) for it in items)
        if len(items) < per_page:
            break
        page += 1
    return nums[:limit]


def fetch_pr(repo: str, number: int) -> tuple[dict, list[dict]]:
    pr = gh_api([f"repos/{repo}/pulls/{number}"])
    files = gh_api(["--paginate", f"repos/{repo}/pulls/{number}/files"])
    return pr, files


def classify_filter(touched: list[str], body_chars: int, max_files: int = 20,
                    min_body: int = 100) -> str | None:
    if not touched:
        return "no_files"
    if body_chars < min_body:
        return "body_too_short"
    if len(touched) > max_files:
        return "too_many_files"
    is_doc = all(DOC_RE.match(f) for f in touched)
    if is_doc:
        return "docs_only"
    is_ci = all(CI_RE.match(f) for f in touched)
    if is_ci:
        return "ci_only"
    is_test = all(TEST_RE.search(f) for f in touched)
    if is_test:
        return "tests_only"
    has_code = any(Path(f).suffix.lower() in CODE_EXTS for f in touched)
    if not has_code:
        return "no_code_files"
    return None  # passes filter


def process_pr(repo: str, number: int, clone_dir: Path) -> PRQrel | None:
    try:
        pr, files = fetch_pr(repo, number)
    except Exception as e:
        # Common: large diffs that the API truncates; just skip
        return None

    touched = [f["filename"] for f in files]
    body = pr.get("body") or ""
    title = pr.get("title") or ""
    problem_statement = (title + "\n\n" + body).strip()

    filter_reason = classify_filter(touched, len(body))

    # Drop tests-only entries from gold_files to test for source-file localization;
    # but still record them under all_files_exist
    if clone_dir.exists():
        missing = [f for f in touched if not (clone_dir / f).exists()]
    else:
        missing = list(touched)
    all_exist = len(missing) == 0 and len(touched) > 0

    patch_loc = sum(int(f.get("additions", 0)) + int(f.get("deletions", 0)) for f in files)

    return PRQrel(
        query_id=f"{repo.replace('/', '__')}-pr-{number}",
        repo=repo,
        problem_statement=problem_statement,
        gold_files=touched,
        base_commit=pr.get("base", {}).get("sha", "") or "",
        created_at=pr.get("created_at", "") or "",
        merged_at=pr.get("merged_at", "") or "",
        difficulty="",
        patch_loc=patch_loc,
        n_gold_files=len(touched),
        all_files_exist=all_exist,
        missing_files=missing,
        filter_reason=filter_reason,
    )


def process_repo(repo: str, merged_start: str, merged_end: str | None,
                 max_per_repo: int,
                 clone_root: Path) -> tuple[str, list[PRQrel]]:
    clone_dir = clone_root / repo.replace("/", "__")
    window = f"{merged_start}..{merged_end}" if merged_end else f">={merged_start}"
    print(f"  [{repo}] listing PRs (merged:{window}, max={max_per_repo})...",
          file=sys.stderr)
    try:
        nums = list_pr_numbers(repo, merged_start, merged_end, max_per_repo)
    except Exception as e:
        print(f"  [{repo}] list failed: {e}", file=sys.stderr)
        return repo, []
    print(f"  [{repo}] processing {len(nums)} PRs...", file=sys.stderr)
    out: list[PRQrel] = []
    for n in nums:
        rec = process_pr(repo, n, clone_dir)
        if rec is not None:
            out.append(rec)
    return repo, out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clone-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--merged-start", default="2026-01-01",
                   help="Lower bound on PR merged_at (inclusive, YYYY-MM-DD)")
    p.add_argument("--merged-end", default=None,
                   help="Optional upper bound on PR merged_at (inclusive, YYYY-MM-DD). "
                        "If provided, query becomes merged:START..END (range); "
                        "else merged:>=START (open-ended).")
    p.add_argument("--cutoff", dest="merged_start", default=argparse.SUPPRESS,
                   help="(Deprecated alias for --merged-start; kept for backward compat.)")
    p.add_argument("--repos", default=None,
                   help="Comma-separated repo subset (e.g. 'django/django,sympy/sympy'). "
                        "Default: all built-in repos.")
    p.add_argument("--max-per-repo", type=int, default=100,
                   help="Process at most N most-recent PRs per repo within the window")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel repos (each repo serialized PR fetches)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    repos = [r.strip() for r in args.repos.split(",")] if args.repos else REPOS

    results: dict[str, list[PRQrel]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(process_repo, r, args.merged_start, args.merged_end,
                        args.max_per_repo, args.clone_root): r
            for r in repos
        }
        for fut in as_completed(futs):
            repo, recs = fut.result()
            results[repo] = recs
            kept = sum(1 for r in recs if r.filter_reason is None)
            print(f"  [{repo}] DONE: {len(recs)} fetched, {kept} pass filter", file=sys.stderr)

    summary: dict[str, Any] = {
        "merged_start": args.merged_start,
        "merged_end": args.merged_end,
        "max_per_repo": args.max_per_repo,
        "repos": repos,
        "per_repo": {},
        "filter_reason_counts": collections.Counter(),
    }

    print(f"\n{'repo':<32s}  {'fetched':>7s}  {'kept':>5s}  {'all_ok':>6s}  reasons",
          file=sys.stderr)
    print("-" * 90, file=sys.stderr)

    total_fetched = 0
    total_kept = 0
    total_all_ok = 0

    for repo in repos:
        recs = results.get(repo, [])
        out_path = args.out_dir / f"{repo.replace('/', '__')}.jsonl"
        with out_path.open("w") as f:
            for rec in recs:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

        reason_counts = collections.Counter(
            r.filter_reason for r in recs if r.filter_reason is not None
        )
        kept = sum(1 for r in recs if r.filter_reason is None)
        all_ok = sum(1 for r in recs if r.filter_reason is None and r.all_files_exist)

        summary["per_repo"][repo] = {
            "n_fetched": len(recs),
            "n_kept": kept,
            "n_all_files_exist_among_kept": all_ok,
            "filter_reason_counts": dict(reason_counts),
            "out_path": str(out_path),
        }
        for k, v in reason_counts.items():
            summary["filter_reason_counts"][k] += v

        total_fetched += len(recs)
        total_kept += kept
        total_all_ok += all_ok

        top_reasons = ", ".join(f"{k}:{v}" for k, v in reason_counts.most_common(3))
        print(f"{repo:<32s}  {len(recs):>7d}  {kept:>5d}  {all_ok:>6d}  {top_reasons}",
              file=sys.stderr)

    print("-" * 90, file=sys.stderr)
    print(f"{'TOTAL':<32s}  {total_fetched:>7d}  {total_kept:>5d}  {total_all_ok:>6d}",
          file=sys.stderr)
    print(f"\nFilter reason totals: {dict(summary['filter_reason_counts'])}", file=sys.stderr)

    summary["filter_reason_counts"] = dict(summary["filter_reason_counts"])
    summary["totals"] = {
        "n_fetched": total_fetched,
        "n_kept": total_kept,
        "n_all_files_exist_among_kept": total_all_ok,
    }
    summary_path = args.out_dir / "post_cutoff_qrels_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[written] {summary_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
