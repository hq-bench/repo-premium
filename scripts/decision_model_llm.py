#!/usr/bin/env python3
"""decision_model_llm.py — zero-shot "Jev-style" LLM decision model.

Same preregistration, action set, dev/test split and LOCKBOX as
decision_model.py (DeepSWE and the R9 pools are never exported here). The
model sees the query plus the top-5 file paths and a short snippet of each
file from two result lists, labelled neutrally "List 1"/"List 2" in a
per-query seeded random order (no retriever names, no hint about the
hypothesis), and returns a typed choice {"action": "list1"|"list2"|"merge"}.
"merge" maps to rrf(dense, bm25). temperature=0; responses cached.

  export  (local)  -> decision_model_llm_prompts.jsonl (dev + test)
  judge   (GPU host)    -> decision_model_llm_cache.jsonl   (needs proxy env)
  analyze (local)  -> decision_model_llm.json
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

ROOT = _HERE.parent
ANALYSIS = DATA_ROOT / "analysis"
PROMPTS = ANALYSIS / "decision_model_llm_prompts.jsonl"
CACHE = ANALYSIS / "decision_model_llm_cache.jsonl"
MODEL = "qwen35-122B"

SYSTEM = ("You help a developer find the source files relevant to a request in a "
          "software repository. Two different search systems were run on the same "
          "repository with the same request. Decide which result list is more likely "
          "to rank the relevant files near the top, or whether merging both lists is "
          "better. Answer only with the JSON object requested.")
PROMPT = """Request:
{query}

List 1 (top 5):
{list1}

List 2 (top 5):
{list2}

Return JSON: {{"action": "list1" | "list2" | "merge"}}"""
SCHEMA = {"type": "object", "properties": {"action": {"type": "string",
          "enum": ["list1", "list2", "merge"]}}, "required": ["action"]}
PROMPT_HASH = hashlib.sha256((SYSTEM + PROMPT).encode()).hexdigest()[:12]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def snippet(text: str) -> str:
    lines = [l.rstrip() for l in text.split("\n") if l.strip()][:8]
    return "\n".join(lines)[:400]


def fmt(paths, snips) -> str:
    out = []
    for i, p in enumerate(paths[:5], 1):
        s = snips.get(p, "")
        out.append(f"{i}. {p}\n" + "\n".join("   " + l for l in s.split("\n")))
    return "\n".join(out)


# ---------------------------------------------------------------- export
def cmd_export(args) -> int:
    from decision_model import load_dev, load_test  # noqa: E402
    from router_ceiling import CANON, QRELS  # noqa: E402
    # raw lists for dev/test (load_* returns features only)
    out = []
    dev_lists = {}
    for cell, pool in CANON.items():
        for repo_dir in sorted(p for p in pool.iterdir() if p.is_dir()):
            for r in read_jsonl(repo_dir / "smoke_test_django.jsonl"):
                dev_lists[r["query_id"]] = (repo_dir.name, r)
    queries = {}
    for cell in CANON:
        for f in QRELS[cell].glob("*.jsonl"):
            for q in read_jsonl(f):
                queries[q["query_id"]] = q.get("problem_statement") or q.get("title", "")
    from run_retrieval import strip_pr_boilerplate  # noqa: E402
    for qid, (repo, r) in sorted(dev_lists.items()):
        snips = {}
        for p in dict.fromkeys(r["dense_top10"][:5] + r["bm25_top10"][:5]):
            fp = args.clones / repo / p
            snips[p] = snippet(fp.read_text(errors="ignore")) if fp.exists() else ""
        out.append(build("dev", qid, strip_pr_boilerplate(queries[qid]), r, snips))
    tau = {t["query_id"]: t for t in read_jsonl(ANALYSIS / "decision_model_tauhat_scaleup.jsonl")}
    test_rows = {t["query_id"] for t in load_test(ANALYSIS / "decision_model_tauhat_scaleup.jsonl")}
    from decision_model import SCALEUP  # noqa: E402
    manifest = {m["task_id"]: m for m in read_jsonl(SCALEUP / "splits_v2" / "retrieval_tasks_v2.jsonl")}
    retr = SCALEUP / "experiment_v1" / "retrieval" / "jina-0.5b"
    for qid in sorted(test_rows):
        task, arm = qid.split("::")
        r = next(x for x in read_jsonl(retr / task / "smoke_test_django.jsonl") if x["query_id"] == qid)
        m = manifest[task]
        q = m["curated_query"] if arm == "curated" else m["raw_query"]
        out.append(build("test", qid, strip_pr_boilerplate(q), r, tau[qid]["snippets"]))
    PROMPTS.write_text("".join(json.dumps(o) + "\n" for o in out))
    print(f"wrote {len(out)} prompts ({sum(o['split']=='dev' for o in out)} dev) -> {PROMPTS}")
    return 0


def build(split, qid, query, r, snips) -> dict:
    rng = random.Random(int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 2**32)
    order = ["dense", "bm25"]
    rng.shuffle(order)
    lists = {"dense": r["dense_top10"], "bm25": r["bm25_top10"]}
    text = PROMPT.format(query=query[:4000], list1=fmt(lists[order[0]], snips),
                         list2=fmt(lists[order[1]], snips))
    return {"split": split, "query_id": qid, "list1": order[0], "list2": order[1],
            "text": text}


# ---------------------------------------------------------------- judge (GPU host)
def parse(content: str) -> str | None:
    m = re.search(r"\{[^{}]*\"action\"[^{}]*\}", content or "", re.S)
    if not m:
        return None
    try:
        a = json.loads(m.group(0)).get("action")
    except json.JSONDecodeError:
        return None
    return a if a in ("list1", "list2", "merge") else None


def call(base, key, text, max_tokens) -> dict:
    body = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": text}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "decide", "schema": SCHEMA}}}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json",
                                          "User-Agent": "coreb-decision-model/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.load(resp)
    msg = out["choices"][0]["message"]
    return {"content": msg.get("content") or "", "usage": out.get("usage", {}),
            "finish_reason": out["choices"][0].get("finish_reason")}


def cmd_judge(args) -> int:
    base, key = os.environ["OPENAI_API_BASE"], os.environ["OPENAI_API_KEY"]
    prompts = read_jsonl(args.prompts)
    done = set()
    if args.cache.exists():
        done = {r["query_id"] for r in read_jsonl(args.cache)
                if r.get("action") and r.get("prompt_hash") == PROMPT_HASH}
    todo = [p for p in prompts if p["query_id"] not in done]
    print(f"{len(todo)} calls ({len(done)} cached)", flush=True)
    lock = threading.Lock()

    def work(p):
        rec = {"query_id": p["query_id"], "prompt_hash": PROMPT_HASH, "model": MODEL}
        for _ in range(4):
            try:
                r = call(base, key, p["text"], args.max_tokens)
                rec.update(r, action=parse(r["content"]))
                if rec["action"]:
                    break
            except Exception as e:  # noqa: BLE001
                rec.update(error=repr(e)[:300], action=None)
        with lock, args.cache.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return bool(rec["action"])

    ok = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for i, g in enumerate(ex.map(work, todo), 1):
            ok += g
            if i % 50 == 0:
                print(f"{i}/{len(todo)} ok={ok}", flush=True)
    print(f"done {ok}/{len(todo)}")
    return 0


# ---------------------------------------------------------------- analyze
def cmd_analyze(args) -> int:
    from decision_model import (ACTIONS, ANALYSIS as A2, boot_repo_task, load_dev,
                                load_test, best_action)  # noqa: E402
    prompts = {p["query_id"]: p for p in read_jsonl(PROMPTS)}
    cache = {}
    for r in read_jsonl(CACHE):
        if r.get("action") and r.get("prompt_hash") == PROMPT_HASH:
            cache[r["query_id"]] = r["action"]
    rng = np.random.default_rng(0)
    rep = {"model": MODEL, "prompt_hash": PROMPT_HASH}
    for split, rows in (("dev", load_dev(args.clones)),
                        ("test", load_test(A2 / "decision_model_tauhat_scaleup.jsonl"))):
        acts, missing = [], 0
        for r in rows:
            a = cache.get(r["query_id"])
            p = prompts[r["query_id"]]
            if a is None:
                missing += 1
                acts.append("dense")
            else:
                acts.append("rrf" if a == "merge" else p[a])
        dense = np.array([r["dense"] for r in rows])
        real = np.array([r[a] for r, a in zip(rows, acts)])
        orc = np.array([best_action(r) for r in rows])
        ch = np.array([ACTIONS.index(a) for a in acts])
        gain = real - dense
        rep[split] = {"n": len(rows), "missing_answers_counted_as_dense": missing,
                      "routed_minus_dense": float(gain.mean()),
                      "routed_minus_dense_ci": boot_repo_task(rows, gain, rng),
                      "oracle_minus_dense": float(np.mean(
                          [max(r[x] for x in ACTIONS) for r in rows]) - dense.mean()),
                      "action_dist": {a: float(np.mean([x == a for x in acts])) for a in ACTIONS},
                      "accuracy_vs_oracle_action": float((orc == ch).mean())}
        if split == "test":
            for scope in ("curated", "raw"):
                idx = [i for i, r in enumerate(rows) if r["cell"] == scope]
                rep[split][scope] = float(gain[idx].mean())
    (ANALYSIS / "decision_model_llm.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--clones", type=Path, default=DATA_ROOT / "repos")
    j = sub.add_parser("judge")
    j.add_argument("--prompts", type=Path, default=PROMPTS)
    j.add_argument("--cache", type=Path, default=CACHE)
    j.add_argument("--workers", type=int, default=8)
    j.add_argument("--max-tokens", type=int, default=8192)
    a = sub.add_parser("analyze")
    a.add_argument("--clones", type=Path, default=DATA_ROOT / "repos")
    args = ap.parse_args()
    return {"export": cmd_export, "judge": cmd_judge, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
