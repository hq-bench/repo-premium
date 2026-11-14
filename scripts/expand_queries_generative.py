#!/usr/bin/env python3
"""Generative lexical expansion: turn a behavioural query into lexical anchors.

Reviewer conjecture this tests: lexical retrieval on a natural-language query
is really an LLM doing the text-to-code mapping, with BM25 left to perform a
code-to-code hard match. If that is right, an LLM that emits identifiers,
paths and hypothetical code before BM25 should recover much of the dense
advantage -- and it should do so by raising query distinctiveness tau, moving
the query rightwards along the premium-decay curve rather than breaking it.

The query set is derived from the canonical per-query result files rather than
re-run through `load_qrels` sampling, so the expanded arm is evaluated on
exactly the queries the cached baseline used. Cell B boilerplate is stripped
here, using the retrieval script's own stripper, so the generator sees exactly
the text BM25 sees.

Emits augmented qrel files carrying the original fields plus:
  expanded_query   stripped original + generated anchors  (arm 2)
  generated_query  generated anchors only                 (arm 3)
  generation       the raw structured fields, for hallucination accounting
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_retrieval import strip_pr_boilerplate  # noqa: E402

FIELDS = (
    "identifiers", "file_or_module_hints", "apis",
    "error_terms", "search_phrases", "hypothetical_code",
)

SYSTEM_PROMPT = """\
You help a code search engine locate the source files that must be edited to \
resolve a software issue. You will be shown an issue description from a real \
repository. You do not have access to the repository.

Emit the literal lexical material a keyword search engine would need to find \
the relevant files. Prefer terms likely to appear verbatim in the source code. \
Do not restate the issue in prose, do not explain your reasoning, and do not \
propose a fix in words.

Return ONLY a JSON object with exactly these keys:
  "identifiers":         list of likely function/class/variable/constant names
  "file_or_module_hints":list of likely file paths or module names
  "apis":                list of likely qualified calls or API signatures
  "error_terms":         list of literal error/warning/log strings involved
  "search_phrases":      list of short literal code phrases worth grepping
  "hypothetical_code":   a short plausible code snippet from the file to change

Every list may be empty. Keep each list to at most 12 entries. Invent nothing \
that the issue text does not support; if you are unsure of a name, omit it \
rather than guessing a plausible-looking one."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_generation(text: str) -> dict[str, Any]:
    """Extract the JSON object, tolerating fenced or prose-wrapped output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model output")
    obj = json.loads(match.group(0))
    out: dict[str, Any] = {}
    for field in FIELDS:
        value = obj.get(field)
        if field == "hypothetical_code":
            out[field] = value if isinstance(value, str) else ""
        elif isinstance(value, list):
            out[field] = [str(v) for v in value if isinstance(v, (str, int, float))][:12]
        else:
            out[field] = []
    return out


def anchors_text(generation: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in FIELDS:
        value = generation.get(field)
        if field == "hypothetical_code":
            if value:
                parts.append(str(value))
        elif value:
            parts.append(" ".join(str(v) for v in value))
    return "\n".join(parts).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True,
                        help="Canonical per-query results defining the query set.")
    parser.add_argument("--qrel-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--strip-boilerplate", action="store_true", default=True,
                    help="Strip PR-template boilerplate (default: on, as in the paper)")
    parser.add_argument("--no-strip-boilerplate", dest="strip_boilerplate", action="store_false")
    parser.add_argument("--model", default="qwen35-122B")
    parser.add_argument("--api-url", default=os.environ.get("OPENAI_API_BASE"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    # Optional headers for endpoints protected by Cloudflare Access.
    parser.add_argument("--cf-client-id",
                        default=os.environ.get("CF_ACCESS_CLIENT_ID"))
    parser.add_argument("--cf-client-secret",
                        default=os.environ.get("CF_ACCESS_CLIENT_SECRET"))
    parser.add_argument("--cf-access-token",
                        default=os.environ.get("CF_ACCESS_TOKEN"),
                        help="Short-lived JWT from `cloudflared access token`.")
    parser.add_argument("--extra-body", default=None,
                        help="JSON dict merged into the request body. Needed for "
                             "locally served hybrid-reasoning models: pass "
                             '\'{"chat_template_kwargs": {"enable_thinking": false}}\' '
                             "to hold a Qwen3 checkpoint in non-thinking mode, so the "
                             "prompt stays byte-identical to the cached generators and "
                             "reasoning tokens cannot exhaust --max-tokens.")
    parser.add_argument("--max-input-chars", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 = all queries.")
    args = parser.parse_args()
    if not args.api_url:
        parser.error("set OPENAI_API_BASE or pass --api-url")
    if not args.api_key:
        parser.error("set OPENAI_API_KEY or pass --api-key")
    extra_kwargs: dict[str, Any] = {}
    if args.extra_body:
        try:
            extra_kwargs["extra_body"] = json.loads(args.extra_body)
        except json.JSONDecodeError as exc:
            parser.error(f"--extra-body is not valid JSON: {exc}")

    headers: dict[str, str] = {}
    if args.cf_client_id and args.cf_client_secret:
        headers["CF-Access-Client-Id"] = args.cf_client_id
        headers["CF-Access-Client-Secret"] = args.cf_client_secret
    elif args.cf_access_token:
        headers["cf-access-token"] = args.cf_access_token
    if headers:
        print(f"[auth] sending Cloudflare Access headers: "
              f"{sorted(headers)}", file=sys.stderr)

    from openai import OpenAI
    client = OpenAI(base_url=args.api_url, api_key=args.api_key,
                    timeout=args.timeout,
                    default_headers=headers or None)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"n_queries": 0, "n_ok": 0, "n_failed": 0, "failed_ids": []}

    for repo_dir in sorted(p for p in args.results_root.iterdir() if p.is_dir()):
        record_path = repo_dir / "smoke_test_django.jsonl"
        qrel_path = args.qrel_dir / f"{repo_dir.name}.jsonl"
        if not record_path.is_file() or not qrel_path.is_file():
            continue
        wanted = [json.loads(l)["query_id"] for l in record_path.read_text().splitlines() if l.strip()]
        by_id = {r["query_id"]: r for r in read_jsonl(qrel_path)}

        out_rows: list[dict[str, Any]] = []
        for qid in wanted:
            row = by_id.get(qid)
            if row is None:
                continue
            if args.limit and stats["n_queries"] >= args.limit:
                break
            stats["n_queries"] += 1
            original = row.get("problem_statement") or ""
            if args.strip_boilerplate:
                original = strip_pr_boilerplate(original)

            generation, error = None, ""
            for attempt in range(1, args.attempts + 1):
                try:
                    response = client.chat.completions.create(
                        model=args.model, temperature=0, max_tokens=args.max_tokens,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": original[:args.max_input_chars]},
                        ],
                        **extra_kwargs,
                    )
                    generation = parse_generation(response.choices[0].message.content or "")
                    break
                except Exception as exc:  # noqa: BLE001 - proxy/parse errors both retryable
                    error = f"{type(exc).__name__}: {exc}"[:300]
                    if attempt < args.attempts:
                        time.sleep(3 * attempt)

            new_row = dict(row)
            new_row["problem_statement_stripped"] = original
            if generation is None:
                stats["n_failed"] += 1
                stats["failed_ids"].append(qid)
                new_row["generation_error"] = error
                # Fall back to the original text so the arm stays evaluable and
                # the failure is visible rather than silently dropping a query.
                new_row["expanded_query"] = original
                new_row["generated_query"] = original
            else:
                stats["n_ok"] += 1
                anchors = anchors_text(generation)
                new_row["generation"] = generation
                new_row["expanded_query"] = (original + "\n" + anchors).strip()
                new_row["generated_query"] = anchors or original
            out_rows.append(new_row)

            if stats["n_queries"] % 25 == 0:
                print(f"  ... {stats['n_queries']} queries "
                      f"({stats['n_ok']} ok, {stats['n_failed']} failed)", flush=True)

        if out_rows:
            (args.out_dir / f"{repo_dir.name}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows)
            )
            print(f"[{repo_dir.name}] wrote {len(out_rows)} expanded queries", flush=True)

    (args.out_dir / "expansion_summary.json").write_text(
        json.dumps({"model": args.model, **stats}, indent=2) + "\n"
    )
    print(json.dumps({"model": args.model, **stats}, indent=2))
    return int(stats["n_failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
