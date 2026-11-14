"""Cell C builder — LLM-rewrite symbol-named queries as behavior-only NL.

Takes the symbol-named bucket from R7 (queries with n_repo_hits >= 3) and asks an
LLM to rewrite each PR description as a brief behavior-only natural-language query
with every identifier, file path, error class, and ticket reference stripped.

This bulks up the "graph-unservable" bucket (the cleanest semantic-only regime
from R7 / Finding 16) without re-mining new PRs.

Default model is gemma-3-1b via the project LiteLLM proxy. Override with --model.

Output (--out): per-rewrite JSONL
    {query_id, repo, cell, gold_files, original_ps, rewritten,
     n_identifiers_orig, n_identifiers_rewritten, accepted}

Usage:
  export OPENAI_API_KEY=<your-litellm-key>
  python scripts/build_cell_c.py \
      --strat-jsonl ${REPO_PREMIUM_DATA:-data}/r7_strat_pooled.jsonl \
      --qrel-cellA-dir ${REPO_PREMIUM_DATA:-data}/coreb_repo_qrels/qrels \
      --qrel-cellB-dir ${REPO_PREMIUM_DATA:-data}/coreb_repo_qrels/qrels_post_cutoff \
      --out ${REPO_PREMIUM_DATA:-data}/coreb_repo_cell_c.jsonl \
      --n 50 --model gemma-3-1b-it
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Reuse the same regex+filter used by graph_search.py so validation is faithful.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_search import extract_identifiers  # noqa: E402


REWRITE_PROMPT = """\
Rewrite the following GitHub PR description as a short, behavior-only natural-language query suitable for code search.

STRICT CONSTRAINTS:
- Strip EVERY function name, class name, method name, file path, module path, error/exception class, decorator name, identifier
- Strip ticket numbers, PR numbers, GitHub URLs, repo names, commit hashes
- Keep only what the change DOES in terms of observable behavior, intent, or symptom
- Output 1-2 sentences, plain prose, no code, no bullets, no headers
- Do NOT use any token that contains an underscore or CamelCase
- Do NOT name files or modules

PR description:
\"\"\"
{ps}
\"\"\"

Behavior-only query (1-2 sentences, plain English, no identifiers):"""


def call_litellm(model: str, prompt: str, api_url: str, api_key: str,
                 temperature: float = 0.2, timeout: int = 60,
                 max_tokens: int = 256) -> str:
    """Single chat completion via OpenAI-compatible LiteLLM proxy."""
    # Lazy import so the script can be inspected without openai installed.
    import openai

    client = openai.OpenAI(base_url=api_url, api_key=api_key, timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def load_qrels(qrel_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in qrel_dir.glob("*.jsonl"):
        for line in p.open():
            r = json.loads(line)
            qid = r.get("query_id")
            if qid:
                out[qid] = r
    return out


def repo_short_to_full(short: str) -> str:
    """Reverse the abbreviation used in build_code_graph etc."""
    # Just match against the keys we know.
    table = {
        "django": "django__django",
        "sympy": "sympy__sympy",
        "sklearn": "scikit-learn__scikit-learn",
        "mpl": "matplotlib__matplotlib",
        "xarray": "pydata__xarray",
        "astropy": "astropy__astropy",
        "pytest": "pytest-dev__pytest",
        "sphinx": "sphinx-doc__sphinx",
        "pylint": "pylint-dev__pylint",
        "requests": "psf__requests",
    }
    return table.get(short, short)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strat-jsonl", type=Path,
                    default=(DATA_ROOT / 'r7_strat_pooled.jsonl'))
    ap.add_argument("--qrel-cellA-dir", type=Path,
                    default=(DATA_ROOT / 'release_src/qrels'))
    ap.add_argument("--qrel-cellB-dir", type=Path,
                    default=(DATA_ROOT / 'release_src/qrels_post_cutoff'))
    ap.add_argument("--out", type=Path,
                    default=(DATA_ROOT / 'coreb_repo_cell_c.jsonl'))
    ap.add_argument("--n", type=int, default=50,
                    help="Number of queries to rewrite.")
    ap.add_argument("--model", default="gemma-4-31B-it",
                    help="OpenAI-compatible model identifier (default: gemma-4-31B-it, "
                         "must be available at your configured endpoint).")
    ap.add_argument("--api-url", default=os.environ.get("OPENAI_API_BASE"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--min-orig-hits", type=int, default=3,
                    help="Only rewrite queries with at least this many repo hits.")
    ap.add_argument("--max-rewritten-ids", type=int, default=2,
                    help="Accept rewrite only if identifier count <= this.")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--resume", action="store_true",
                    help="Skip query_ids already in --out.")
    args = ap.parse_args()
    if not args.api_url or not args.api_key:
        ap.error("set OPENAI_API_BASE and OPENAI_API_KEY (or --api-url and --api-key)")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load stratified data, pick symbol-named candidates.
    candidates = []
    for line in args.strat_jsonl.open():
        q = json.loads(line)
        if q.get("n_repo_hits", 0) >= args.min_orig_hits:
            candidates.append(q)
    candidates.sort(key=lambda q: (-q.get("n_repo_hits", 0), q["_key"]))
    print(f"[cell-c] {len(candidates)} candidates with >= {args.min_orig_hits} hits",
          file=sys.stderr)

    # 2) Pull problem_statements from qrels (lazy load per cell).
    qrel_A = load_qrels(args.qrel_cellA_dir)
    qrel_B = load_qrels(args.qrel_cellB_dir)
    print(f"[cell-c]   qrel pool: A={len(qrel_A)}  B={len(qrel_B)}",
          file=sys.stderr)

    # 3) Resume from existing output.
    done: set[str] = set()
    if args.resume and args.out.exists():
        for line in args.out.open():
            try:
                done.add(json.loads(line)["query_id"])
            except Exception:
                pass
        print(f"[cell-c]   resume: {len(done)} already done", file=sys.stderr)

    # 4) Iterate, rewrite, validate, append.
    out_fh = args.out.open("a" if args.resume else "w")
    n_done = 0
    n_accept = 0
    n_reject = 0
    start = time.time()
    for q in candidates:
        if n_done >= args.n:
            break
        # Recover the raw query_id (strip cell prefix from r7_strat key).
        raw_qid = q["_key"].split("__", 1)[1]
        if raw_qid in done:
            continue
        cell = q["_cell"]
        qrel_row = (qrel_A if cell == "A" else qrel_B).get(raw_qid)
        if not qrel_row:
            continue
        ps = qrel_row.get("problem_statement", "").strip()
        if not ps:
            continue
        n_orig_ids = len(extract_identifiers(ps))
        prompt = REWRITE_PROMPT.format(ps=ps[:6000])
        try:
            rewritten = call_litellm(
                args.model, prompt, args.api_url, args.api_key,
                temperature=args.temperature, max_tokens=args.max_tokens)
        except Exception as e:
            print(f"[cell-c] LLM call failed for {raw_qid}: {e}",
                  file=sys.stderr)
            continue
        n_new_ids = len(extract_identifiers(rewritten))
        accepted = n_new_ids <= args.max_rewritten_ids
        rec = {
            "query_id": raw_qid + "_c",  # Cell C namespace
            "source_query_id": raw_qid,
            "cell_origin": cell,
            "repo": qrel_row.get("repo"),
            "gold_files": qrel_row.get("gold_files"),
            "base_commit": qrel_row.get("base_commit"),
            "original_problem_statement": ps,
            "problem_statement": rewritten,
            "n_identifiers_original": n_orig_ids,
            "n_identifiers_rewritten": n_new_ids,
            "accepted": accepted,
            "rewriter_model": args.model,
        }
        out_fh.write(json.dumps(rec) + "\n")
        out_fh.flush()
        n_done += 1
        if accepted:
            n_accept += 1
        else:
            n_reject += 1
        if n_done % 5 == 0:
            elapsed = time.time() - start
            print(f"[cell-c]   {n_done} done  "
                  f"({n_accept} accepted, {n_reject} rejected)  "
                  f"{elapsed:.1f}s  rate={n_done/elapsed:.2f}/s",
                  file=sys.stderr)

    out_fh.close()
    n = max(n_done, 1)
    print(f"\n[cell-c] wrote {args.out}", file=sys.stderr)
    print(f"  total           = {n_done}", file=sys.stderr)
    print(f"  accepted        = {n_accept} ({n_accept/n*100:.1f}%)",
          file=sys.stderr)
    print(f"  rejected        = {n_reject}", file=sys.stderr)
    print(f"  elapsed         = {time.time()-start:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
