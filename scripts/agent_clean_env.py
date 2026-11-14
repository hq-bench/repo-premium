#!/usr/bin/env python3
"""Contamination-free task environments for the agent evaluation.

Two leaks made the first agent-eval runs unsound, and neither was closed by
blocking the network:

1. `clone_heldout_repos.py` clones each repository `--depth 200` at HEAD, and
   `git worktree add --detach <base>` shares that object store. The working
   tree was pinned at the base commit but `.git` was not, so `git log --all`,
   `git branch -a` and `git show <sha>` reached the merged fix offline. On the
   held-out sample every one of the 30 tasks had post-base commits locally
   reachable, and agents read the gold commit in 18 runs.
2. The retriever-wrapper condition was only a sentence in the system prompt
   asking the model not to use `grep`/`find`/`rg`. Agents ignored it in 20-25
   of 30 trajectories per arm, so the "restricted" arms were not restricted.

`materialize_clean` fixes (1) by exporting the base tree with `git archive`
into a fresh repository whose entire history is one synthetic commit; nothing
about the future exists in it. Scoring is unchanged, since it reads
`git diff HEAD --name-only` plus untracked files.

`install_search_blockers` fixes (2) by putting refusing stubs for the search
commands first on PATH, including a `git` shim that passes everything through
except `git grep`. Absolute paths and in-language file walks remain possible,
so `audit_search_bypass` records them rather than pretending they cannot
happen.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

BLOCKED = ("grep", "egrep", "fgrep", "rg", "ripgrep", "find", "ag", "ack")

_STUB = """#!/bin/sh
echo "{name}: disabled in this condition - use the provided retriever tool" >&2
exit 1
"""

_GIT_STUB = """#!/bin/sh
# Pass through to real git, except the search subcommand.
for a in "$@"; do
  case "$a" in
    -*) continue ;;
    grep) echo "git grep: disabled in this condition - use the provided retriever tool" >&2; exit 1 ;;
    *) break ;;
  esac
done
exec {real_git} "$@"
"""


def materialize_clean(repo_dir: str | Path, dest: Path, base_commit: str) -> Path:
    """Export `base_commit`'s tree into `dest` as a fresh single-commit repo."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(repo_dir), "archive", base_commit],
        capture_output=True, check=True,
    )
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)
    env = os.environ | {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    base = ["git", "-C", str(dest), "-c", "user.email=eval@localhost",
            "-c", "user.name=eval", "-c", "commit.gpgsign=false"]
    subprocess.run(base + ["init", "--quiet", "-b", "main"], check=True, env=env)
    subprocess.run(base + ["add", "-A"], check=True, env=env)
    subprocess.run(base + ["commit", "--quiet", "--allow-empty", "-m", "base"],
                   check=True, env=env)
    return dest


def verify_clean(dest: Path) -> dict:
    """Assert the environment exposes exactly one commit and no remotes."""
    def out(*a):
        return subprocess.run(["git", "-C", str(dest), *a],
                              capture_output=True, text=True).stdout.strip()
    n_commits = out("rev-list", "--all", "--count")
    return {
        "n_commits": int(n_commits) if n_commits.isdigit() else -1,
        "n_refs": len([r for r in out("show-ref").splitlines() if r.strip()]),
        "remotes": [r for r in out("remote").splitlines() if r.strip()],
    }


def install_search_blockers(bin_dir: Path) -> Path:
    """Write refusing stubs for the search commands into `bin_dir`."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in BLOCKED:
        p = bin_dir / name
        p.write_text(_STUB.format(name=name))
        p.chmod(0o755)
    real_git = shutil.which("git")
    if real_git is None:
        raise RuntimeError("git must be installed and available on PATH")
    gp = bin_dir / "git"
    gp.write_text(_GIT_STUB.format(real_git=real_git))
    gp.chmod(0o755)
    return bin_dir


_BYPASS = re.compile(
    r"(/usr/bin/|/bin/|/usr/local/bin/)(grep|find|rg|ag|ack)\b"
    r"|\bos\.walk\b|\bglob\.glob\b|\bPath\([^)]*\)\.rglob\b"
    r"|\bgit\s+grep\b"
)
_HISTORY = re.compile(r"git\s+(log|show|branch|rev-list|cat-file|reflog)\b")


def audit_search_bypass(commands: list[str]) -> dict:
    """Record attempts to search around the blockers, and history probes."""
    bypass = [c for c in commands if _BYPASS.search(c or "")]
    history = [c for c in commands if _HISTORY.search(c or "")]
    return {
        "n_search_bypass": len(bypass), "search_bypass": bypass[:10],
        "n_history_probes": len(history), "history_probes": history[:10],
    }
