"""run_agent_eval.py — Batch mini-swe-agent over the held-out 30-query sample.

For each query:
  1. Create a git worktree of the repo at base_commit.
  2. Run mini-swe-agent (non-interactive) inside the worktree with the
     problem_statement as the task.
  3. Diff the worktree against base_commit to find which files the agent touched.
  4. Score precision/recall against gold_files.
  5. Save per-query result + trajectory.

Resumable: if {out_dir}/{qid}.result.json already exists, skip that query.

Usage:
  python3 run_agent_eval.py \
      --sample ${REPO_PREMIUM_DATA:-data}/agent_eval_30.jsonl \
      --out-dir ${REPO_PREMIUM_DATA:-data}/agent_eval_runs/raw_shell_qwen122b \
      --model openai/qwen35-122B \
      --step-limit 25 \
      --timeout 900
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_clean_env import (  # noqa: E402
    audit_search_bypass, materialize_clean, verify_clean,
)

# Overridable: the cluster venv cannot exec on the box that still reaches
# the model proxy, so the rerun uses a matching venv built there.
MINI_BIN = os.environ.get("MINI_SWE_AGENT_BIN",
                          "mini-swe-agent")
# Per-run root: parallel arms would otherwise materialise into the same
# /tmp/<qid> path and delete each other's task trees mid-run.
WORKTREE_ROOT = Path(os.environ.get("AGENT_WORKTREE_ROOT", str(DATA_ROOT / "agent_eval_runs")))


def git(args: list[str], cwd: str | Path | None = None, check: bool = True) -> str:
    res = subprocess.run(
        ["git"] + args,
        cwd=str(cwd) if cwd else None,
        check=check,
        capture_output=True,
        text=True,
    )
    return res.stdout.strip()


def make_worktree(repo_dir: str, qid: str, base_commit: str) -> Path:
    """Materialise the base tree with no history attached.

    A `git worktree` shares the source repository's object store, so the
    agent could reach commits made after `base_commit` -- including the fix --
    with `git log --all` or `git show`, without any network. See
    agent_clean_env for the measurement that forced this change.
    """
    WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)
    return materialize_clean(repo_dir, WORKTREE_ROOT / qid, base_commit)


def cleanup_worktree(repo_dir: str, wt: Path) -> None:
    if wt.exists():
        shutil.rmtree(wt, ignore_errors=True)


def touched_files(wt: Path) -> list[str]:
    """Return paths that changed vs HEAD (modified, added, deleted, untracked)."""
    modified = git(["-C", str(wt), "diff", "HEAD", "--name-only"]).splitlines()
    untracked = git(["-C", str(wt), "ls-files", "--others", "--exclude-standard"]).splitlines()
    return sorted({p.strip() for p in (modified + untracked) if p.strip()})


# Held-out tasks are mined from merged PRs, and backport statements quote the
# upstream PR URL and merge SHA, so a shell agent on a networked node can fetch
# the gold patch. minisweagent's local environment runs each command with
# `os.environ | config.env`, and that merge applies to the AGENT's bash only --
# the harness's own model calls use the parent environment -- so pointing the
# proxy variables at a closed port blocks curl/wget/pip/git-over-http for the
# agent while leaving the LLM client reachable.
BLACKHOLE = "http://127.0.0.1:9"
NETWORK_ENV = {
    "http_proxy": BLACKHOLE, "https_proxy": BLACKHOLE, "all_proxy": BLACKHOLE,
    "HTTP_PROXY": BLACKHOLE, "HTTPS_PROXY": BLACKHOLE, "ALL_PROXY": BLACKHOLE,
    "no_proxy": "", "NO_PROXY": "", "GIT_TERMINAL_PROMPT": "0",
}

NET_CMD = re.compile(
    r"(^|[;&|`$(\s])(curl|wget|nc|ncat|telnet)\s"
    r"|(^|[;&|`$(\s])gh\s+(api|pr|issue|repo|search)"
    r"|git\s+(fetch|pull|clone|ls-remote)"
    r"|pip3?\s+(install|download)"
    r"|\burllib\b|\brequests\.get\b"
)
GOLD_ARTIFACT = re.compile(
    r"https?://[^\s'\"]*"
    r"(api\.github\.com/repos/[^\s'\"]*/pulls?/\d+"
    r"|github\.com/[^\s'\"]+/(pull|commit)/[0-9a-f]+"
    r"|\.diff|\.patch)", re.I)
# Cloning the upstream repository is the same leak by another route: the default
# branch already contains the merged fix, so this counts as a gold fetch even
# though no pull-request URL appears in the command.
CLONE_UPSTREAM = re.compile(r"git\s+clone[^\n]*github\.com", re.I)


def audit_trajectory(traj_path: Path) -> dict:
    """Flag network use by the agent, and fetches of this task's own gold PR.

    Reads `messages[].extra.actions[].command`: assistant `content` is null in
    these trajectories, so an audit that greps `content` silently scans nothing.
    """
    try:
        traj = json.loads(traj_path.read_text())
    except Exception as exc:  # noqa: BLE001 - a missing trajectory is not fatal
        return {"audited": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
    messages = traj.get("messages") or []
    commands = [
        a.get("command", "")
        for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant"
        for a in ((m.get("extra") or {}).get("actions") or [])
    ]
    net = [c for c in commands if NET_CMD.search(c or "")]
    gold = [c for c in net
            if GOLD_ARTIFACT.search(c or "") or CLONE_UPSTREAM.search(c or "")]
    return {
        "audited": True,
        "all_commands": commands,
        "n_commands": len(commands),
        "n_network_commands": len(net),
        "network_commands": net[:20],
        "fetched_gold_artifact": bool(gold),
        "gold_artifact_commands": gold[:10],
    }


def run_agent(model: str, task: str, traj_path: Path, cwd: Path,
              step_limit: int, timeout: int, block_network: bool = True,
              extra_body: str | None = None) -> dict:
    cmd = [
        MINI_BIN,
        "-m", model,
        "-t", task,
        "-y", "--exit-immediately",
        "-o", str(traj_path),
        "-c", "mini.yaml",
        "-c", "model.model_kwargs.max_tokens=8192",
        "-c", "model.model_kwargs.temperature=0.0",
        "-c", "agent.cost_limit=0",
        "-c", f"agent.step_limit={step_limit}",
    ]
    if block_network:
        # Values are JSON-quoted: the config parser reads bare 0 as an int and
        # the environment model requires strings.
        cmd += [a for k, v in NETWORK_ENV.items()
                for a in ("-c", f"environment.env.{k}={json.dumps(v)}")]
    if extra_body:
        cmd += ["-c", f"model.model_kwargs.extra_body={extra_body}"]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
        )
        return {
            "exit_code": proc.returncode,
            "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-2000:],
            "timed_out": False,
            "wall_seconds": round(time.time() - t0, 1),
        }
    except subprocess.TimeoutExpired as e:
        return {
            "exit_code": -1,
            "stdout_tail": (e.stdout or b"")[-3000:].decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")[-3000:],
            "stderr_tail": (e.stderr or b"")[-2000:].decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")[-2000:],
            "timed_out": True,
            "wall_seconds": round(time.time() - t0, 1),
        }


def score(touched: list[str], gold: list[str]) -> dict:
    t = set(touched)
    g = set(gold)
    hit = sorted(t & g)
    precision = len(hit) / len(t) if t else 0.0
    recall = len(hit) / len(g) if g else 0.0
    return {
        "n_touched": len(t),
        "n_gold": len(g),
        "n_hit": len(hit),
        "hit": hit,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "exact_match": t == g,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", type=str, default="openai/qwen35-122B")
    ap.add_argument("--step-limit", type=int, default=25)
    ap.add_argument("--model-extra-body", default=None,
                    help='JSON merged into model.model_kwargs.extra_body, e.g. '
                         '\'{"chat_template_kwargs": {"enable_thinking": false}}\' '
                         "for the within-weight thinking ablation.")
    ap.add_argument("--allow-network", action="store_true",
                    help="Let the agent's shell reach the network. OFF by default: "
                         "backport task statements quote their own upstream PR URL, "
                         "so a networked agent can fetch the gold patch. Proxy "
                         "variables stop well-behaved tools (curl/wget/pip/git over "
                         "http) but not raw sockets, so every run is audited too and "
                         "the audit is written to each result file.")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0, help="Only run first N queries")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in args.sample.read_text().splitlines() if l.strip()]
    if args.limit:
        recs = recs[: args.limit]

    print(f"[start] {len(recs)} queries, model={args.model}, step_limit={args.step_limit}, timeout={args.timeout}s", flush=True)

    for i, rec in enumerate(recs, 1):
        qid = rec["query_id"]
        result_path = args.out_dir / f"{qid}.result.json"
        if result_path.exists():
            print(f"[{i}/{len(recs)}] {qid} — skip (already done)", flush=True)
            continue

        repo_dir = rec["repo_dir"]
        base = rec["base_commit"]
        task_text = rec["problem_statement"][:8000]
        traj_path = args.out_dir / f"{qid}.trajectory.json"

        print(f"[{i}/{len(recs)}] {qid} ...", flush=True)
        try:
            wt = make_worktree(repo_dir, qid, base)
        except subprocess.CalledProcessError as e:
            print(f"  worktree FAILED: {e.stderr}", flush=True)
            continue

        try:
            run_info = run_agent(
                args.model, task_text, traj_path, wt, args.step_limit, args.timeout,
                block_network=not args.allow_network,
                extra_body=args.model_extra_body,
            )
            net_audit = audit_trajectory(traj_path)
            env_check = verify_clean(wt)
            bypass = audit_search_bypass(net_audit.get("all_commands", []))
            if env_check.get("n_commits", -1) != 1 or env_check.get("remotes"):
                print(f"  !! {qid} task environment is not clean: {env_check}", flush=True)
            if net_audit.get("fetched_gold_artifact"):
                print(f"  !! {qid} fetched a gold PR artifact over the network; "
                      f"exclude it or rerun without --allow-network", flush=True)
            touched = touched_files(wt)
            scoring = score(touched, rec["gold_files"])

            result = {
                "query_id": qid,
                "repo": rec["repo"],
                "base_commit": base,
                "model": args.model,
                "step_limit": args.step_limit,
                "touched": touched,
                "gold": rec["gold_files"],
                "network_blocked": not args.allow_network,
                "network_audit": net_audit,
                "clean_env": env_check,
                "search_audit": bypass,
                **scoring,
                **run_info,
            }
            result_path.write_text(json.dumps(result, indent=2))
            print(
                f"  touched={len(touched)} hit={scoring['n_hit']}/{scoring['n_gold']} "
                f"P={scoring['precision']:.2f} R={scoring['recall']:.2f} "
                f"wall={run_info['wall_seconds']}s timeout={run_info['timed_out']}",
                flush=True,
            )
        finally:
            cleanup_worktree(repo_dir, wt)

    # Aggregate summary
    results = []
    for p in sorted(args.out_dir.glob("*.result.json")):
        results.append(json.loads(p.read_text()))
    if results:
        n = len(results)
        precision = sum(r["precision"] for r in results) / n
        recall = sum(r["recall"] for r in results) / n
        any_hit = sum(1 for r in results if r["n_hit"] > 0) / n
        exact = sum(1 for r in results if r["exact_match"]) / n
        timed_out = sum(1 for r in results if r["timed_out"]) / n
        summary = {
            "n_runs": n,
            "model": args.model,
            "avg_precision": round(precision, 4),
            "avg_recall": round(recall, 4),
            "any_hit_rate": round(any_hit, 4),
            "exact_match_rate": round(exact, 4),
            "timeout_rate": round(timed_out, 4),
        }
        (args.out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n[summary] {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
