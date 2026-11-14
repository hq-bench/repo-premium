"""run_agent_eval_wrapper.py — Agent-eval where the LLM is restricted to a
single retriever wrapper (grep|dense|router) built on the held-out repos.

Design:
  - The agent may only invoke `coreb_search 'query text'` for exploration.
  - The runner prepends a CONSTRAINT block to the task text saying so.
  - A tiny shell wrapper at $WORKTREE/coreb_bin/coreb_search dispatches to
    coreb_search.py with the pre-built index and the selected mode.
  - Per-query worktree PATH is augmented to include coreb_bin so the tool is
    callable by name.

Assumes:
  - Per-repo indices live at {index_root}/{repo_dir_name}/.
  - coreb_search.py and its container invocation are on the server side at
    paths configured by COREB_SEARCH_SCRIPT and APPTAINER_IMAGE.

Usage:
  python3 run_agent_eval_wrapper.py \\
      --sample ${REPO_PREMIUM_DATA:-data}/agent_eval_30.jsonl \\
      --index-root ${REPO_PREMIUM_DATA:-data}/agent_eval_wrapper/indices \\
      --out-dir ${REPO_PREMIUM_DATA:-data}/agent_eval_runs/wrapper_dense \\
      --wrapper-mode dense \\
      --model openai/qwen35-122B \\
      --step-limit 30 --timeout 900
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
import os
import shutil
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_clean_env import (  # noqa: E402
    audit_search_bypass, install_search_blockers, materialize_clean, verify_clean,
)

# Overridable: the cluster venv cannot exec on the box that still reaches
# the model proxy, so the rerun uses a matching venv built there.
MINI_BIN = os.environ.get("MINI_SWE_AGENT_BIN",
                          "mini-swe-agent")
COREB_PY = os.environ.get("COREB_SEARCH_SCRIPT", str(Path(__file__).with_name("coreb_search.py")))
SIF_PATH = os.environ.get("APPTAINER_IMAGE", str(DATA_ROOT / "containers/retriever.sif"))
APPTAINER_BIND = os.environ.get("APPTAINER_BIND", f"{DATA_ROOT}:{DATA_ROOT},{Path(__file__).resolve().parents[1]}:{Path(__file__).resolve().parents[1]}")
# The GPU cluster can no longer reach the model proxy, so the rerun drives the
# retriever with a plain CPU interpreter instead of the apptainer image.
RETRIEVER_PY = os.environ.get("COREB_SEARCH_PY", "")
# Per-run root: parallel arms would otherwise materialise into the same
# /tmp/<qid> path and delete each other's task trees mid-run.
WORKTREE_ROOT = Path(os.environ.get("AGENT_WORKTREE_ROOT", str(DATA_ROOT / "agent_eval_runs")))


CONSTRAINT_TMPL = """CONSTRAINT: For all code exploration inside this repository, you MUST use ONLY the pre-installed retriever tool. Do NOT run `find`, `grep`, `rg`, `ag`, `ripgrep`, `ack`, `git grep`, or any other direct file-search or file-listing command. You may still `cat` a specific file once the retriever returns a path.

Invoke the retriever exactly as:
    coreb_search 'natural-language or code query here'

The tool returns a JSON list of hits (top 20) with fields `path`, `start_line`, `end_line`, `score`, `snippet`. Read those, then decide which file(s) to open with `cat`. If your first query does not surface useful results, refine the query text; do NOT bypass the tool.

Retriever mode for this run: {mode}.

TASK:
{task}
"""


def git(args, cwd=None, check=True):
    res = subprocess.run(["git"] + args, cwd=str(cwd) if cwd else None,
                         check=check, capture_output=True, text=True)
    return res.stdout.strip()


def make_worktree(repo_dir, qid, base_commit):
    """Materialise the base tree with no history attached (see
    agent_clean_env): a shared object store let earlier runs read the fix."""
    WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)
    return materialize_clean(repo_dir, WORKTREE_ROOT / qid, base_commit)


def cleanup_worktree(repo_dir, wt):
    if wt.exists():
        shutil.rmtree(wt, ignore_errors=True)


def touched_files(wt):
    modified = git(["-C", str(wt), "diff", "HEAD", "--name-only"]).splitlines()
    untracked = git(["-C", str(wt), "ls-files", "--others",
                     "--exclude-standard"]).splitlines()
    return sorted({p.strip() for p in (modified + untracked) if p.strip()})


def install_wrapper(wt: Path, index_dir: Path, mode: str) -> Path:
    """Write a coreb_search shell wrapper into wt/coreb_bin and return the dir."""
    bin_dir = wt / "coreb_bin"
    bin_dir.mkdir(exist_ok=True)
    wrapper = bin_dir / "coreb_search"
    if RETRIEVER_PY:
        wrapper.write_text(f"""#!/bin/bash
# auto-generated per-query wrapper -- mode={mode} (direct interpreter)
if [ $# -lt 1 ]; then
  echo "usage: coreb_search 'query text'" >&2
  exit 2
fi
exec {shlex.quote(RETRIEVER_PY)} {shlex.quote(COREB_PY)} query \\
  --index-dir {shlex.quote(str(index_dir))} --mode {mode} --k 20 --query "$*"
""")
        wrapper.chmod(0o755)
        install_search_blockers(bin_dir)
        return bin_dir
    wrapper.write_text(f"""#!/bin/bash
# auto-generated per-query wrapper — mode={mode}
if [ $# -lt 1 ]; then
  echo "usage: coreb_search 'query text'" >&2
  exit 2
fi
apptainer exec --nv \\
  --env USE_TF=NO --env TRANSFORMERS_NO_TF=1 \\
  --bind {shlex.quote(APPTAINER_BIND)} \\
  {shlex.quote(SIF_PATH)} \\
  python3 {shlex.quote(COREB_PY)} query \\
    --index-dir {shlex.quote(str(index_dir))} \\
    --mode {mode} \\
    --k 20 \\
    --query "$*"
""")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    # The condition was previously only a sentence in the prompt, and agents
    # ignored it in 20-25 of 30 trajectories per arm. Enforce it on PATH.
    install_search_blockers(bin_dir)
    return bin_dir


def run_agent(model, task, traj_path, cwd, step_limit, timeout, extra_path):
    env = os.environ.copy()
    env["PATH"] = f"{extra_path}:{env.get('PATH', '')}"
    cmd = [
        MINI_BIN, "-m", model, "-t", task,
        "-y", "--exit-immediately",
        "-o", str(traj_path),
        "-c", "mini.yaml",
        "-c", "model.model_kwargs.max_tokens=8192",
        "-c", "model.model_kwargs.temperature=0.0",
        "-c", "agent.cost_limit=0",
        "-c", f"agent.step_limit={step_limit}",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True,
                              text=True, timeout=timeout)
        return {
            "exit_code": proc.returncode,
            "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-2000:],
            "timed_out": False,
            "wall_seconds": round(time.time() - t0, 1),
        }
    except subprocess.TimeoutExpired as e:
        so = (e.stdout or b"")[-3000:]
        se = (e.stderr or b"")[-2000:]
        return {
            "exit_code": -1,
            "stdout_tail": so.decode(errors="replace") if isinstance(so, bytes) else so[-3000:],
            "stderr_tail": se.decode(errors="replace") if isinstance(se, bytes) else se[-2000:],
            "timed_out": True,
            "wall_seconds": round(time.time() - t0, 1),
        }


def score(touched, gold):
    t = set(touched); g = set(gold)
    hit = sorted(t & g)
    p = len(hit) / len(t) if t else 0.0
    r = len(hit) / len(g) if g else 0.0
    return {"n_touched": len(t), "n_gold": len(g), "n_hit": len(hit),
            "hit": hit, "precision": round(p, 4), "recall": round(r, 4),
            "exact_match": t == g}


def repo_name(repo_dir: str) -> str:
    return Path(repo_dir).name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--wrapper-mode", choices=["grep", "dense", "router"],
                    required=True)
    ap.add_argument("--model", default="openai/qwen35-122B")
    ap.add_argument("--step-limit", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in args.sample.read_text().splitlines()
            if l.strip()]
    if args.limit:
        recs = recs[: args.limit]

    print(f"[start] n={len(recs)} model={args.model} mode={args.wrapper_mode}",
          flush=True)

    for i, rec in enumerate(recs, 1):
        qid = rec["query_id"]
        result_path = args.out_dir / f"{qid}.result.json"
        if result_path.exists():
            print(f"[{i}/{len(recs)}] {qid} — skip", flush=True)
            continue

        repo_dir = rec["repo_dir"]
        base = rec["base_commit"]
        rname = repo_name(repo_dir)
        index_dir = args.index_root / rname
        if not (index_dir / "chunks.jsonl").exists():
            print(f"[{i}/{len(recs)}] {qid} — SKIP no index at {index_dir}",
                  flush=True)
            continue

        task_text = CONSTRAINT_TMPL.format(
            mode=args.wrapper_mode, task=rec["problem_statement"][:8000],
        )
        traj_path = args.out_dir / f"{qid}.trajectory.json"

        print(f"[{i}/{len(recs)}] {qid} ...", flush=True)
        try:
            wt = make_worktree(repo_dir, qid, base)
        except subprocess.CalledProcessError as e:
            print(f"  worktree FAILED: {e.stderr}", flush=True)
            continue

        try:
            bin_dir = install_wrapper(wt, index_dir, args.wrapper_mode)
            run_info = run_agent(args.model, task_text, traj_path, wt,
                                 args.step_limit, args.timeout, str(bin_dir))
            touched = touched_files(wt)
            # Exclude the injected coreb_bin dir from touched files
            touched = [p for p in touched if not p.startswith("coreb_bin/")]
            scoring = score(touched, rec["gold_files"])
            env_check = verify_clean(wt)
            cmds = []
            if traj_path.exists():
                try:
                    tj = json.loads(traj_path.read_text())
                    cmds = [a.get("command", "")
                            for m in tj.get("messages", [])
                            if isinstance(m, dict) and m.get("role") == "assistant"
                            for a in ((m.get("extra") or {}).get("actions") or [])]
                except Exception:  # noqa: BLE001 - audit must not break a run
                    cmds = []
            bypass = audit_search_bypass(cmds)

            result = {
                "query_id": qid, "repo": rec["repo"], "base_commit": base,
                "model": args.model, "wrapper_mode": args.wrapper_mode,
                "step_limit": args.step_limit,
                "touched": touched, "gold": rec["gold_files"],
                "clean_env": env_check, "search_audit": bypass,
                **scoring, **run_info,
            }
            result_path.write_text(json.dumps(result, indent=2))
            print(f"  touched={len(touched)} hit={scoring['n_hit']}/{scoring['n_gold']} "
                  f"P={scoring['precision']:.2f} R={scoring['recall']:.2f} "
                  f"wall={run_info['wall_seconds']}s timeout={run_info['timed_out']}",
                  flush=True)
        finally:
            cleanup_worktree(repo_dir, wt)

    results = [json.loads(p.read_text()) for p in sorted(args.out_dir.glob("*.result.json"))]
    if results:
        n = len(results)
        summary = {
            "n_runs": n, "model": args.model,
            "wrapper_mode": args.wrapper_mode,
            "avg_precision": round(sum(r["precision"] for r in results) / n, 4),
            "avg_recall": round(sum(r["recall"] for r in results) / n, 4),
            "any_hit_rate": round(sum(1 for r in results if r["n_hit"] > 0) / n, 4),
            "exact_match_rate": round(sum(1 for r in results if r["exact_match"]) / n, 4),
            "timeout_rate": round(sum(1 for r in results if r["timed_out"]) / n, 4),
        }
        (args.out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n[summary] {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
