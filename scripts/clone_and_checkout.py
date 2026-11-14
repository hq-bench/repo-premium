"""
Clone and pin all CoREB-Repo candidate repositories to a single snapshot.

Two tiers:

**Python (10 repos, anchored to SWE-bench Verified)**
  - Pick the most recent (`created_at` max) SWE-bench Verified instance for that repo
  - Use that instance's `base_commit` as the snapshot SHA
  - This guarantees every issue we eval has a valid `base_commit` ≤ snapshot SHA

**Non-Python (6 repos, manual selection)**
  - Clone main/master HEAD at the time of script execution
  - Record the resolved SHA into the manifest for reproducibility
  - One entry uses a subpath (`kubernetes/kubernetes` → eval only `pkg/scheduler/`)

For every repo the script writes:
  - `{root}/{org__repo}/` ← shallow checkout at the pinned SHA
  - manifest entry: org/repo, lang, snapshot SHA, file count, LOC, ext distribution, subpath

Usage:
  conda activate llm
  python scripts/clone_and_checkout.py \\
      --root ${REPO_PREMIUM_DATA:-data}/repos \\
      --manifest scripts/field_repos_manifest.json \\
      [--only-python | --only-nonpython] \\
      [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Repo shortlist (mirrors docs/coreb_repo_repo_survey.md §4 + decision O1=B)
# ---------------------------------------------------------------------------


@dataclass
class RepoSpec:
    org_repo: str            # "django/django"
    lang: str                # "python" / "javascript" / "typescript" / "go" / "java"
    tier: str                # "swebench-verified" / "manual"
    subpath: str | None = None    # eval only this subpath; None = full repo
    notes: str = ""


PYTHON_REPOS: list[RepoSpec] = [
    RepoSpec("django/django",                 "python", "swebench-verified", notes="cap 50 queries"),
    RepoSpec("sympy/sympy",                   "python", "swebench-verified", notes="cap 50 queries"),
    RepoSpec("sphinx-doc/sphinx",             "python", "swebench-verified"),
    RepoSpec("matplotlib/matplotlib",         "python", "swebench-verified"),
    RepoSpec("scikit-learn/scikit-learn",     "python", "swebench-verified"),
    RepoSpec("astropy/astropy",               "python", "swebench-verified"),
    RepoSpec("pydata/xarray",                 "python", "swebench-verified"),
    RepoSpec("pytest-dev/pytest",             "python", "swebench-verified"),
    RepoSpec("pylint-dev/pylint",             "python", "swebench-verified"),
    RepoSpec("psf/requests",                  "python", "swebench-verified"),
]

NONPYTHON_REPOS: list[RepoSpec] = [
    RepoSpec("facebook/react",                "javascript", "manual"),
    RepoSpec("axios/axios",                   "javascript", "manual"),
    RepoSpec("prettier/prettier",             "javascript", "manual"),
    RepoSpec("gin-gonic/gin",                 "go",         "manual"),
    RepoSpec("spring-projects/spring-boot",   "java",       "manual"),
    RepoSpec("kubernetes/kubernetes",         "go",         "manual", subpath="pkg/scheduler"),
]


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------


@dataclass
class RepoManifestEntry:
    org_repo: str
    lang: str
    tier: str
    subpath: str | None
    snapshot_sha: str
    clone_path: str
    file_count: int
    total_loc: int
    ext_distribution: dict[str, int] = field(default_factory=dict)
    swebench_instance_count: int = 0
    swebench_pinned_instance_id: str | None = None
    notes: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# SWE-bench commit resolution
# ---------------------------------------------------------------------------


def resolve_python_snapshots() -> dict[str, tuple[str, str, int]]:
    """For each Python repo, return (base_commit, instance_id, n_instances_in_repo).

    Picks the SWE-bench Verified instance with the **latest** `created_at` per repo
    so all earlier issues still have a valid base_commit ≤ snapshot SHA.
    """
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    by_repo: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in ds:
        by_repo[row["repo"]].append(row)

    out: dict[str, tuple[str, str, int]] = {}
    for repo, rows in by_repo.items():
        # Sort by created_at (lexicographic ISO works); take the most recent
        rows_sorted = sorted(rows, key=lambda r: r.get("created_at") or "")
        last = rows_sorted[-1]
        out[repo] = (last["base_commit"], last["instance_id"], len(rows))
    return out


# ---------------------------------------------------------------------------
# Git ops
# ---------------------------------------------------------------------------


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def clone_full(repo_url: str, dest: Path) -> None:
    """Full clone — needed when we must check out an arbitrary historical SHA."""
    if dest.exists():
        print(f"  [skip clone] {dest} already exists", file=sys.stderr)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [clone] {repo_url} → {dest}", file=sys.stderr)
    run(["git", "clone", repo_url, str(dest)])


def clone_shallow(repo_url: str, dest: Path) -> None:
    """Shallow clone — fine for non-Python repos pinned to HEAD."""
    if dest.exists():
        print(f"  [skip clone] {dest} already exists", file=sys.stderr)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [clone --depth 1] {repo_url} → {dest}", file=sys.stderr)
    run(["git", "clone", "--depth", "1", repo_url, str(dest)])


def checkout(dest: Path, sha: str) -> None:
    print(f"  [checkout] {dest.name} @ {sha[:12]}", file=sys.stderr)
    run(["git", "checkout", "--detach", sha], cwd=dest)


def head_sha(dest: Path) -> str:
    return run(["git", "rev-parse", "HEAD"], cwd=dest).stdout.strip()


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


# File extensions of interest per language (used to scope grep/BM25/dense indexing later)
LANG_EXTS = {
    "python":     {".py"},
    "javascript": {".js", ".jsx", ".mjs", ".cjs"},
    "typescript": {".ts", ".tsx"},
    "go":         {".go"},
    "java":       {".java"},
}


def is_source_file(path: Path, lang: str) -> bool:
    exts = LANG_EXTS.get(lang, set())
    if not exts:
        return False
    return path.suffix.lower() in exts


def inventory_repo(root: Path, lang: str, subpath: str | None) -> dict[str, Any]:
    """Walk the repo (or subpath) and count files/LOC/extensions.

    Ignores `.git/`, virtualenvs, node_modules, build artifacts.
    """
    walk_root = root / subpath if subpath else root
    if not walk_root.exists():
        return {"file_count": 0, "total_loc": 0, "ext_distribution": {}}

    IGNORE_PARTS = {
        ".git", "node_modules", ".tox", ".venv", "venv", "__pycache__",
        ".pytest_cache", "build", "dist", "target", ".gradle", ".idea",
        ".cache", "vendor",
    }

    n_files = 0
    n_loc = 0
    ext_counter: collections.Counter[str] = collections.Counter()

    for p in walk_root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in IGNORE_PARTS for part in p.parts):
            continue
        # Count both source-file inventory and broader extension stats
        ext = p.suffix.lower() or "<noext>"
        ext_counter[ext] += 1
        if is_source_file(p, lang):
            n_files += 1
            try:
                with p.open("rb") as f:
                    # Cheap LOC: count newlines
                    n_loc += sum(1 for _ in f)
            except OSError:
                pass

    return {
        "file_count": n_files,
        "total_loc": n_loc,
        "ext_distribution": dict(ext_counter.most_common(20)),
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def process_repo(
    spec: RepoSpec,
    root: Path,
    swebench_snapshots: dict[str, tuple[str, str, int]] | None,
    dry_run: bool,
) -> RepoManifestEntry:
    repo_url = f"https://github.com/{spec.org_repo}.git"
    dest = root / spec.org_repo.replace("/", "__")

    print(f"\n[{spec.org_repo}] tier={spec.tier} lang={spec.lang}", file=sys.stderr)

    entry = RepoManifestEntry(
        org_repo=spec.org_repo,
        lang=spec.lang,
        tier=spec.tier,
        subpath=spec.subpath,
        snapshot_sha="",
        clone_path=str(dest),
        file_count=0,
        total_loc=0,
        notes=spec.notes,
    )

    try:
        if spec.tier == "swebench-verified":
            if swebench_snapshots is None:
                raise RuntimeError("swebench_snapshots was not loaded but spec.tier=swebench-verified")
            sha, instance_id, n_inst = swebench_snapshots[spec.org_repo]
            entry.swebench_pinned_instance_id = instance_id
            entry.swebench_instance_count = n_inst
            entry.snapshot_sha = sha

            if not dry_run:
                clone_full(repo_url, dest)
                checkout(dest, sha)
        else:  # manual / non-Python
            if not dry_run:
                clone_shallow(repo_url, dest)
                entry.snapshot_sha = head_sha(dest)
            else:
                entry.snapshot_sha = "<HEAD at run time>"

        if not dry_run:
            inv = inventory_repo(dest, spec.lang, spec.subpath)
            entry.file_count = inv["file_count"]
            entry.total_loc = inv["total_loc"]
            entry.ext_distribution = inv["ext_distribution"]

    except subprocess.CalledProcessError as e:
        entry.error = f"git failed: {e.stderr.strip()}"
        print(f"  [ERROR] {entry.error}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        entry.error = f"{type(e).__name__}: {e}"
        print(f"  [ERROR] {entry.error}", file=sys.stderr)

    return entry


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="parent dir for all repo clones")
    p.add_argument("--manifest", type=Path, required=True, help="output manifest JSON path")
    p.add_argument("--only-python", action="store_true")
    p.add_argument("--only-nonpython", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print plan only, no git ops")
    args = p.parse_args()

    if args.only_python and args.only_nonpython:
        p.error("--only-python and --only-nonpython are mutually exclusive")

    args.root.mkdir(parents=True, exist_ok=True)

    # Pick which specs to process
    specs: list[RepoSpec] = []
    if not args.only_nonpython:
        specs += PYTHON_REPOS
    if not args.only_python:
        specs += NONPYTHON_REPOS

    swebench_snapshots: dict[str, tuple[str, str, int]] | None = None
    if any(s.tier == "swebench-verified" for s in specs):
        print("[load] SWE-bench Verified for base_commit resolution", file=sys.stderr)
        swebench_snapshots = resolve_python_snapshots()
        print(f"[load] resolved snapshots for {len(swebench_snapshots)} repos", file=sys.stderr)

    entries: list[RepoManifestEntry] = []
    for spec in specs:
        entries.append(process_repo(spec, args.root, swebench_snapshots, args.dry_run))

    manifest = {
        "schema_version": 1,
        "root": str(args.root),
        "n_repos": len(entries),
        "n_python": sum(1 for e in entries if e.lang == "python"),
        "n_nonpython": sum(1 for e in entries if e.lang != "python"),
        "n_failed": sum(1 for e in entries if e.error),
        "repos": [asdict(e) for e in entries],
    }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n[written] {args.manifest}", file=sys.stderr)

    # Summary table
    print()
    print(f"{'org/repo':40s}  {'lang':12s}  {'sha':12s}  {'files':>7s}  {'loc':>10s}")
    for e in entries:
        sha = (e.snapshot_sha or "—")[:12]
        status = " ✗" if e.error else "  "
        print(f"{status}{e.org_repo:38s}  {e.lang:12s}  {sha:12s}  "
              f"{e.file_count:7d}  {e.total_loc:10d}")

    if manifest["n_failed"]:
        print(f"\n⚠  {manifest['n_failed']} repo(s) failed; inspect manifest for details",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
