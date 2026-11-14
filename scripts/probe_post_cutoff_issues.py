"""
Probe each CoREB-Repo repo for post-cutoff merged PRs.

Goal: estimate how many query/file pairs we can mine *after* a held-out cutoff
(default 2026-01-01) to use as the contamination-controlled subset for the
Embedding Premium law.

For each repo we:
  1. Query the GitHub Search API for merged PRs with `merged:>=CUTOFF`.
  2. Report total_count.
  3. Fetch the first 10 PRs and report titles + #files changed as a sanity check.

Usage:
  gh auth status   # required
  conda run -n llm python scripts/probe_post_cutoff_issues.py \\
      [--cutoff 2026-01-01] \\
      [--out scripts/post_cutoff_probe.json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


REPOS = [
    # Python tier (SWE-bench Verified)
    "django/django",
    "sympy/sympy",
    "sphinx-doc/sphinx",
    "matplotlib/matplotlib",
    "scikit-learn/scikit-learn",
    "astropy/astropy",
    "pydata/xarray",
    "pytest-dev/pytest",
    "pylint-dev/pylint",
    "psf/requests",
    # Non-Python tier (manual)
    # NOTE: facebook/react was renamed to react/react in 2025 (Meta org move).
    # GitHub REST redirects but Search API does not, so use canonical name.
    "react/react",
    "axios/axios",
    "prettier/prettier",
    "gin-gonic/gin",
    "spring-projects/spring-boot",
    "kubernetes/kubernetes",
]


@dataclass
class RepoProbe:
    org_repo: str
    total_merged_post_cutoff: int
    sample_titles: list[str]
    sample_pr_numbers: list[int]
    error: str | None = None


def gh_api(args: list[str]) -> dict:
    """Run `gh api <args>` and return parsed JSON. Raises on failure."""
    cmd = ["gh", "api", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"gh api failed: {' '.join(cmd)}\n{result.stderr}")
    return json.loads(result.stdout)


def probe_repo(repo: str, cutoff: str) -> RepoProbe:
    """Count post-cutoff merged PRs and grab a small title sample."""
    query = f"repo:{repo} is:pr is:merged merged:>={cutoff}"
    try:
        # search/issues endpoint also returns PRs (PR is a subtype of issue on GitHub).
        # per_page=10 to get a small sample without burning the rate limit.
        data = gh_api([
            "-X", "GET", "search/issues",
            "-f", f"q={query}",
            "-f", "per_page=10",
            "-f", "sort=created",
            "-f", "order=desc",
        ])
        total = int(data.get("total_count", 0))
        items = data.get("items", [])
        return RepoProbe(
            org_repo=repo,
            total_merged_post_cutoff=total,
            sample_titles=[it.get("title", "")[:100] for it in items[:5]],
            sample_pr_numbers=[int(it.get("number", 0)) for it in items[:5]],
        )
    except Exception as e:
        return RepoProbe(
            org_repo=repo,
            total_merged_post_cutoff=0,
            sample_titles=[],
            sample_pr_numbers=[],
            error=str(e),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", default="2026-01-01",
                        help="ISO date; count merged PRs with merged>=cutoff")
    parser.add_argument("--out", type=Path,
                        default=Path("scripts/post_cutoff_probe.json"))
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel API workers")
    args = parser.parse_args()

    print(f"[probe] cutoff={args.cutoff}, repos={len(REPOS)}", file=sys.stderr)

    results: list[RepoProbe] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_repo = {pool.submit(probe_repo, r, args.cutoff): r for r in REPOS}
        for fut in as_completed(future_to_repo):
            res = fut.result()
            results.append(res)
            tag = "ERR" if res.error else "OK "
            print(f"  [{tag}] {res.org_repo:40s} merged_PRs(>={args.cutoff}) = "
                  f"{res.total_merged_post_cutoff}", file=sys.stderr)

    # Sort by total desc for readable output
    results.sort(key=lambda r: -r.total_merged_post_cutoff)

    payload = {
        "cutoff": args.cutoff,
        "n_repos": len(REPOS),
        "total_merged_prs": sum(r.total_merged_post_cutoff for r in results),
        "by_repo": [asdict(r) for r in results],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[written] {args.out}", file=sys.stderr)

    print(f"\n=== Summary (cutoff >= {args.cutoff}) ===")
    print(f"{'repo':<40s}  {'merged_PRs':>10s}")
    print("-" * 55)
    for r in results:
        suffix = f"  ERROR: {r.error}" if r.error else ""
        print(f"{r.org_repo:<40s}  {r.total_merged_post_cutoff:>10d}{suffix}")
    print("-" * 55)
    print(f"{'TOTAL':<40s}  {payload['total_merged_prs']:>10d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
