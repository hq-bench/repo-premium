#!/usr/bin/env python3
"""Audit of unlabeled retrieved files: are the 'misses' really irrelevant?

Gold sets are the files one resolving change edited, so other files relevant to the change
are scored as misses. This audit asks an LLM judge whether retrieved non-gold files are
relevant, separately for dense retrieval and tuned (whole-file) BM25, to see (i) how often
the labels miss relevant files and (ii) whether that bias favors one retriever.

  build   (local) for every Raw-Post query and an equal-size seeded sample of Cur-Pre queries,
          take the top-3 non-gold files of the canonical dense run and of whole-file BM25,
          plus one gold file per query as a blind positive control; write items with the
          stripped query and the file text (head, capped) to audit_items.jsonl
  judge   (venv with openai) qwen35-122B, temperature 0, one JSON verdict per item:
          EDIT (would plausibly be edited for this or an equally valid change),
          CONTEXT (useful to read, not edited) or NO
  analyze (local) rates by pool, retriever and item type

Labels are the paper's (indexed, non-test files); queries come from coreb_repo_data_indexed.
"""
from __future__ import annotations
from paths import DATA_ROOT

import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = DATA_ROOT / 'analysis/indexed/relevance_audit'
MAX_CHARS = 6000
MODEL = "qwen35-122B"
PROMPT = """You are reviewing a code-search result for a software repository.

Change description (from an issue or pull request):
<<<
{query}
>>>

Candidate file: {path}
File contents (possibly truncated):
<<<
{text}
>>>

Question: would a developer implementing the described change plausibly need to EDIT this
file (as part of this change or an equally valid way of making it)? If not, would they need
to READ it to understand or make the change?

Answer with a JSON object only: {{"verdict": "EDIT" | "CONTEXT" | "NO", "reason": "<one sentence>"}}"""


def build() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_retrieval import strip_pr_boilerplate
    SM = DATA_ROOT / 'smoke_test_results'
    repos = DATA_ROOT / "repos"
    pools = {"Cur-Pre": ("canon202607/jina05_cellA_c1000", "bm25_sweep202610/bm25_cellA_c1000000000", "qrels"),
             "Raw-Post": ("canon202607/jina05_cellB_c1000", "bm25_sweep202610/bm25_cellB_c1000000000", "qrels_post_cutoff")}
    items, rng = [], random.Random(20261011)
    n_b = None
    for pool, (dd, bd, qd) in pools.items():
        Q = {}
        for f in (DATA_ROOT / 'release_src' / qd).glob("*.jsonl"):
            for l in f.read_text().splitlines():
                r = json.loads(l); Q[r["query_id"]] = r["problem_statement"]
        recs = []
        for f in sorted((SM / dd).glob("*/smoke_test_django.jsonl")):
            repo = f.parent.name
            bm = {json.loads(l)["query_id"]: json.loads(l) for l in (SM / bd / repo / "smoke_test_django.jsonl").read_text().splitlines()}
            for l in f.read_text().splitlines():
                r = json.loads(l)
                if r["query_id"] in bm:
                    recs.append((repo, r, bm[r["query_id"]]))
        if pool == "Raw-Post":
            n_b = len(recs)
        else:
            recs = rng.sample(recs, min(len(recs), 132))
        for repo, r, b in recs:
            gold = set(r["gold_files"])
            q = strip_pr_boilerplate(Q.get(r["query_id"], ""))[:4000]
            cands = {}
            for ret, top in (("dense", r["dense_top10"]), ("bm25", b["bm25_top10"])):
                for rank, p in enumerate([x for x in top if x not in gold][:3]):
                    cands.setdefault(p, []).append(f"{ret}@{rank + 1}")
            cands[rng.choice(sorted(gold))] = ["gold"]
            for p, tags in cands.items():
                try:
                    text = (repos / repo / p).read_text(errors="ignore")[:MAX_CHARS]
                except OSError:
                    continue
                items.append({"id": f"{r['query_id']}::{p}", "pool": pool, "repo": repo, "query_id": r["query_id"],
                              "path": p, "tags": tags, "query": q, "text": text})
    OUT.mkdir(parents=True, exist_ok=True)
    rng.shuffle(items)
    (OUT / "audit_items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    print(len(items), "items;", n_b, "Raw-Post queries")


def judge(path: str) -> None:
    from openai import OpenAI
    client = OpenAI(base_url=os.environ["OPENAI_API_BASE"].rstrip("/") + "/v1", api_key=os.environ["OPENAI_API_KEY"])
    items = [json.loads(l) for l in open(path)]
    out_p = Path(path).with_name("audit_verdicts.jsonl")
    done = {json.loads(l)["id"] for l in open(out_p)} if out_p.exists() else set()
    with open(out_p, "a") as fh:
        for i, it in enumerate(items):
            if it["id"] in done:
                continue
            msg = PROMPT.format(query=it["query"], path=it["path"], text=it["text"])
            verdict, raw = None, ""
            for _ in range(3):
                try:
                    resp = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=300,
                                                          messages=[{"role": "user", "content": msg}],
                                                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                    raw = resp.choices[0].message.content or ""
                    s = raw[raw.find("{"): raw.rfind("}") + 1]
                    verdict = json.loads(s).get("verdict")
                    if verdict in ("EDIT", "CONTEXT", "NO"):
                        break
                except Exception as e:  # noqa: BLE001
                    raw = f"ERROR {e}"
            fh.write(json.dumps({"id": it["id"], "verdict": verdict, "raw": raw[:500]}) + "\n"); fh.flush()
            if i % 50 == 0:
                print(i, len(items), flush=True)


def analyze() -> None:
    items = {json.loads(l)["id"]: json.loads(l) for l in open(OUT / "audit_items.jsonl")}
    ver = {json.loads(l)["id"]: json.loads(l)["verdict"] for l in open(OUT / "audit_verdicts.jsonl")}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_source_only_mirror import is_test
    res = {}
    for pool in ("Cur-Pre", "Raw-Post"):
        res[pool] = {}
        groups = {"gold (control)": lambda t: "gold" in t,
                  "dense non-gold top-3": lambda t: any(x.startswith("dense") for x in t),
                  "BM25 non-gold top-3": lambda t: any(x.startswith("bm25") for x in t),
                  "dense only": lambda t: any(x.startswith("dense") for x in t) and not any(x.startswith("bm25") for x in t),
                  "BM25 only": lambda t: any(x.startswith("bm25") for x in t) and not any(x.startswith("dense") for x in t)}
        for name, sel in groups.items():
            vs = [(ver.get(k), is_test(it["path"])) for k, it in items.items() if it["pool"] == pool and sel(it["tags"]) and ver.get(k)]
            n = len(vs)
            if not n:
                continue
            res[pool][name] = {"n": n, **{v: round(sum(x == v for x, _ in vs) / n, 3) for v in ("EDIT", "CONTEXT", "NO")},
                               "edit_nontest": round(sum(x == "EDIT" and not t for x, t in vs) / n, 3), "test_share": round(sum(t for _, t in vs) / n, 3)}
        print(pool, json.dumps(res[pool], indent=1))
    (OUT / "audit_summary.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    {"build": build, "judge": lambda: judge(sys.argv[2]), "analyze": analyze}[sys.argv[1]]()
