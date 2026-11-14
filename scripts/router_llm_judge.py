#!/usr/bin/env python3
"""router_llm_judge.py — D2: can a model that reads the full query text
predict when BM25 beats dense?

D1 (router_ceiling.py) showed that query-only structural features, tau-hat
and length score at chance. D2 swaps the hand-built features for readers of
the whole query:

  * zero-shot LLM judges (qwen35-122B, gemma-4-31B-it), shown ONLY the query
    text and a neutral description of the two retrievers, returning a
    constrained JSON choice plus p(lexical);
  * query embeddings from the canonical encoder (jina-code-0.5b) and a
    general text encoder (Qwen3-Embedding-0.6B) + logistic regression under
    leave-one-repository-out CV.

Same 612 canonical Cell A+B queries, labels (BM25 strictly beats dense,
jina-code-0.5b, c=1000), LOO-repo protocol and repo-clustered bootstrap as
D1; the D1 loaders are imported, not copied.

Three subcommands, because the LLM proxy is only reachable from the GPU host:

  export   (local)  write the query file the judge reads
  judge    (GPU host)    call the proxy; every response is cached, reruns are free.
                    Stdlib only; reads OPENAI_API_BASE / OPENAI_API_KEY from env.
  embed    (GPU host)    query embeddings -> .npy (the encoders are cached there)
  analyze  (local)  metrics -> router_llm_judge.json

Run export/analyze in the `llm` conda env.
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import threading
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
ANALYSIS = DATA_ROOT / "analysis"
QUERIES_OUT = ANALYSIS / "router_llm_judge_queries.jsonl"
CACHE = ANALYSIS / "router_llm_judge_cache.jsonl"
MAX_QUERY_CHARS = 8000
MODELS = ["qwen35-122B", "gemma-4-31B-it"]

# Neutral by design: no mention of rare tokens, identifiers, tau, or which
# retriever we expect to win on what.
SYSTEM = ("You are helping choose a search engine for a software repository. "
          "Answer with JSON only.")
PROMPT = """A developer's request (an issue or pull-request description) will be used as a search query to find the source files that must be changed to resolve it. Two search engines are available over the repository's source code:

- "lexical": keyword matching (BM25). It ranks code chunks by how many of the query's words occur in them, weighting words that are uncommon in the repository more heavily.
- "semantic": a neural embedding model. It maps the query and each code chunk to vectors and ranks chunks by similarity of meaning, even when they share few words.

Which engine will rank the files that need to change higher for this query?

Query:
<<<
{query}
>>>

Respond with a JSON object: {{"retriever": "lexical" or "semantic", "p_lexical": probability from 0 to 1 that "lexical" is the better engine}}."""
PROMPT_HASH = hashlib.sha256((SYSTEM + PROMPT).encode()).hexdigest()[:12]
SCHEMA = {"type": "object",
          "properties": {"retriever": {"type": "string", "enum": ["lexical", "semantic"]},
                         "p_lexical": {"type": "number", "minimum": 0, "maximum": 1}},
          "required": ["retriever", "p_lexical"]}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------- export
def cmd_export(args) -> int:
    sys.path.insert(0, str(_HERE))
    from router_ceiling import CANON, QRELS, load_pool  # noqa: E402
    from run_retrieval import strip_pr_boilerplate  # noqa: E402
    rows = []
    for cell, pool in CANON.items():
        queries = {}
        for f in QRELS[cell].glob("*.jsonl"):
            for r in read_jsonl(f):
                queries[r["query_id"]] = r.get("problem_statement") or r.get("title", "")
        for r in load_pool(pool):
            text = strip_pr_boilerplate(queries[r["query_id"]])[:MAX_QUERY_CHARS]
            rows.append({"cell": cell, "repo": r["repo"], "query_id": r["query_id"],
                         "dense": r["dense"], "bm25": r["bm25"], "text": text})
    QUERIES_OUT.parent.mkdir(parents=True, exist_ok=True)
    QUERIES_OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} queries -> {QUERIES_OUT}")
    return 0


# ---------------------------------------------------------------- judge
def parse_answer(content: str) -> dict | None:
    m = re.search(r"\{[^{}]*\"retriever\"[^{}]*\}", content or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if obj.get("retriever") not in ("lexical", "semantic"):
        return None
    try:
        p = float(obj.get("p_lexical"))
    except (TypeError, ValueError):
        return None
    return {"retriever": obj["retriever"], "p_lexical": min(max(p, 0.0), 1.0)}


def call(base: str, key: str, model: str, query: str, max_tokens: int) -> dict:
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": PROMPT.format(query=query)}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "route", "schema": SCHEMA}}}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 # Cloudflare rejects urllib's default User-Agent (403/1010)
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json",
                                          "User-Agent": "coreb-router-judge/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.load(resp)
    msg = out["choices"][0]["message"]
    return {"content": msg.get("content") or "", "usage": out.get("usage", {}),
            "finish_reason": out["choices"][0].get("finish_reason")}


def cmd_judge(args) -> int:
    base, key = os.environ["OPENAI_API_BASE"], os.environ["OPENAI_API_KEY"]
    queries = read_jsonl(args.queries)
    done = set()
    if args.cache.exists():
        for r in read_jsonl(args.cache):
            if r.get("answer") is not None:
                done.add((r["model"], r["query_id"], r["prompt_hash"]))
    lock = threading.Lock()
    todo = [(m, q) for m in args.models for q in queries
            if (m, q["query_id"], PROMPT_HASH) not in done]
    print(f"{len(todo)} calls to make ({len(done)} cached)", flush=True)

    def work(item):
        model, q = item
        rec = {"model": model, "query_id": q["query_id"], "prompt_hash": PROMPT_HASH}
        for attempt in range(3):
            try:
                r = call(base, key, model, q["text"], args.max_tokens)
                rec.update(r, answer=parse_answer(r["content"]))
                if rec["answer"] is not None:
                    break
            except Exception as e:  # noqa: BLE001 — record and retry
                rec.update(error=repr(e)[:300], answer=None)
        with lock, args.cache.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec["answer"] is not None

    ok = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for i, good in enumerate(ex.map(work, todo), 1):
            ok += good
            if i % 50 == 0:
                print(f"{i}/{len(todo)} ok={ok}", flush=True)
    print(f"done: {ok}/{len(todo)} parsed")
    return 0


# ---------------------------------------------------------------- embed
def embed_path(folder: Path, model_name: str) -> Path:
    return folder / f"router_llm_judge_emb_{model_name.replace('/', '__')}.npy"


def embed(texts: list[str], model_name: str, device: str):
    # Plain query encoding, no instruction prefix, as smoke_test_django.py does.
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    return enc.encode(texts, batch_size=8, normalize_embeddings=True, convert_to_numpy=True)


def cmd_embed(args) -> int:
    import numpy as np
    X = embed([q["text"] for q in read_jsonl(args.queries)], args.embed_model, args.device)
    out = embed_path(args.out_dir, args.embed_model)
    np.save(out, X)
    print(f"{X.shape} -> {out}")
    return 0


# ---------------------------------------------------------------- analyze
def cmd_analyze(args) -> int:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0, str(_HERE))
    from router_ceiling import auc_or_none, clustered_ci, loo_predict, loo_route  # noqa: E402

    data = read_jsonl(args.queries)
    groups = np.array([x["repo"] for x in data])
    cells = np.array([x["cell"] for x in data])
    dense = np.array([x["dense"] for x in data])
    bm25 = np.array([x["bm25"] for x in data])
    y = (bm25 > dense).astype(float)
    nontie = dense != bm25
    oracle_choice = np.where(nontie, y, np.nan)  # 1 = BM25 better, nan = tie

    def summarize(p, routed, hard=None) -> dict:
        res = {}
        for scope, mask in [("pooled", np.ones(len(y), bool)),
                            ("cellA", cells == "cellA"), ("cellB", cells == "cellB")]:
            g, yy, pp, nt = groups[mask], y[mask], p[mask], nontie[mask]
            dd, rr = dense[mask], routed[mask]
            r = {"auroc_nontie": auc_or_none(yy[nt], pp[nt]),
                 "auroc_nontie_ci": clustered_ci(g, lambda i: auc_or_none(yy[i][nt[i]], pp[i][nt[i]])),
                 "dense": dd.mean(), "routed": rr.mean(),
                 "routed_minus_dense": (rr - dd).mean(),
                 "routed_minus_dense_ci": clustered_ci(g, lambda i: (rr[i] - dd[i]).mean()),
                 "frac_routed_to_bm25": float((rr != dd).mean())}
            if hard is not None:
                hh = hard[mask]
                hr = np.where(hh == 1, bm25[mask], dd)
                oc = oracle_choice[mask]
                r["hard_frac_lexical"] = float(hh.mean())
                r["hard_routed_minus_dense"] = (hr - dd).mean()
                r["hard_routed_minus_dense_ci"] = clustered_ci(g, lambda i: (hr[i] - dd[i]).mean())
                r["hard_agreement_with_oracle_nontie"] = float((hh[nt] == oc[nt]).mean())
            res[scope] = r
        return res

    report = {"n": int(len(y)), "n_bm25_win": int(y.sum()), "n_nontie": int(nontie.sum()),
              "prompt_hash": PROMPT_HASH, "judges": {}, "embedding": {}}

    # ---- LLM judges
    cache = read_jsonl(args.cache) if args.cache.exists() else []
    qid_idx = {x["query_id"]: i for i, x in enumerate(data)}
    for model in args.models:
        ans = {}
        for r in cache:
            if r["model"] == model and r["prompt_hash"] == PROMPT_HASH and r.get("answer"):
                ans[r["query_id"]] = r["answer"]
        cov = len(ans)
        if cov < len(data):
            report["judges"][model] = {"coverage": cov, "status": "incomplete"}
            if cov == 0:
                continue
        mask = np.array([x["query_id"] in ans for x in data])
        p = np.array([ans[x["query_id"]]["p_lexical"] if m else 0.0 for x, m in zip(data, mask)])
        hard = np.array([float(ans[x["query_id"]]["retriever"] == "lexical") if m else 0.0
                         for x, m in zip(data, mask)])
        # tuned threshold: the judge's p is a fixed (untrained) score, so the
        # only thing learned is theta, tuned on training repos' realised gains.
        routed = dense.copy()
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            gains = {t: ((bm25[tr] - dense[tr]) * (p[tr] >= t)).sum() for t in np.unique(p[tr])}
            theta, gain = max(gains.items(), key=lambda kv: kv[1])
            if gain > 0:
                routed[te] = np.where(p[te] >= theta, bm25[te], dense[te])
        out = summarize(p, routed, hard)
        out["coverage"] = cov
        out["p_lexical_distinct_values"] = int(len(np.unique(p)))
        report["judges"][model] = out

    # ---- embedding + logreg (LOO-repo, nested theta via loo_route)
    for model_name in args.embed_models:
        X = np.load(embed_path(ANALYSIS, model_name)) if embed_path(ANALYSIS, model_name).exists() \
            else embed([x["text"] for x in data], model_name, args.device)
        for C in (0.1, 1.0):
            make = lambda C=C: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C))
            p = loo_predict(X, y, groups, make)
            routed = loo_route(p, dense, bm25, groups, X, y, make)
            report["embedding"][f"{model_name}/logreg_C{C}"] = summarize(p, routed)

    args.out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps(report, indent=1, default=lambda v: round(float(v), 4)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    j = sub.add_parser("judge")
    j.add_argument("--queries", type=Path, default=QUERIES_OUT)
    j.add_argument("--cache", type=Path, default=CACHE)
    j.add_argument("--models", nargs="+", default=MODELS)
    j.add_argument("--workers", type=int, default=8)
    j.add_argument("--max-tokens", type=int, default=8192)
    e = sub.add_parser("embed")
    e.add_argument("--queries", type=Path, default=QUERIES_OUT)
    e.add_argument("--embed-model", default="jinaai/jina-code-embeddings-0.5b")
    e.add_argument("--device", default="cuda")
    e.add_argument("--out-dir", type=Path, default=ANALYSIS)
    a = sub.add_parser("analyze")
    a.add_argument("--queries", type=Path, default=QUERIES_OUT)
    a.add_argument("--cache", type=Path, default=CACHE)
    a.add_argument("--models", nargs="+", default=MODELS)
    a.add_argument("--embed-models", nargs="+",
                   default=["jinaai/jina-code-embeddings-0.5b", "Qwen/Qwen3-Embedding-0.6B"])
    a.add_argument("--device", default="mps")
    a.add_argument("--out", type=Path, default=ANALYSIS / "router_llm_judge.json")
    args = ap.parse_args()
    return {"export": cmd_export, "judge": cmd_judge, "embed": cmd_embed,
            "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
