#!/usr/bin/env python3
"""Fidelity check for the alias-control arm (review #14).

The alias arm is only a semantic control if the aliases actually preserve
meaning. Two checks:

  1. A seeded sample of 50 aliased queries written side by side (original
     stripped text / alias text / term mapping) for a human spot check.
  2. An automatic judge over EVERY alias: shown the original report and one
     term -> replacement pair, does the replacement mean the same thing in
     context? The judge is a different model family from the generator
     (generator qwen35-122B, judge gemma-4-31B-it) so the check is not
     self-grading.

Reads qrels_cell{A,B}_alias/ written by anchor_alias_control.py. Runs on the GPU host
(the only host that reaches the proxy); responses cached per query.
"""

from __future__ import annotations
from paths import DATA_ROOT

import argparse
import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_alias_control import chat, parse_json_object, surface_forms  # noqa: E402

PROMPT_VERSION = "fidelity-v1"
GENERATOR = "qwen35-122B"

SYSTEM = "You check word replacements in software issue reports. Answer with one JSON object only."

USER_TMPL = """Issue report:
<<<
{text}
>>>

Each line below proposes replacing a term from the report with a new term. For EACH line, decide whether a developer reading the report with the replacement would understand exactly the same thing as with the original term, in the context of this report. Minor stylistic awkwardness is fine; a change of meaning, a vaguer or broader meaning, or a different concept is not.

{pairs}

Answer with a JSON object mapping each original term exactly as listed to true (meaning preserved) or false (meaning changed), e.g. {{"parse_header": true, "because": false}}."""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA_ROOT)
    ap.add_argument("--judge", default="gemma-4-31B-it")
    ap.add_argument("--n-sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-query-chars", type=int, default=8000)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    args = ap.parse_args()

    items = []  # (cell, repo, row, audit)
    for cell in ("A", "B"):
        d = args.data / "ablation" / f"qrels_cell{cell}_alias"
        for audit_path in sorted(d.glob("*_audit.jsonl")):
            repo = audit_path.name[: -len("_audit.jsonl")]
            rows = {r["query_id"]: r for r in map(json.loads, (d / f"{repo}.jsonl")
                                                  .read_text().splitlines()) if r}
            for a in map(json.loads, audit_path.read_text().splitlines()):
                if a["query_id"] in rows:
                    items.append((cell, repo, rows[a["query_id"]], a))
    print(f"[load] {len(items)} aliased queries", file=sys.stderr)

    cache = {}
    if args.cache.exists():
        for l in args.cache.read_text().splitlines():
            c = json.loads(l)
            cache[c["key"]] = c

    def judge(item):
        cell, repo, row, a = item
        text = row["problem_statement_stripped"]
        pairs = {(surface_forms(text, t) or [t])[0]: v
                 for t, v in a["aliases"].items() if v}
        if not pairs:
            return item, {}
        key = hashlib.sha256(json.dumps([PROMPT_VERSION, args.judge, row["query_id"],
                                         sorted(pairs.items())]).encode()).hexdigest()
        if key not in cache:
            prompt = USER_TMPL.format(text=text[:args.max_query_chars], pairs="\n".join(
                f'- "{o}" -> "{n}"' for o, n in pairs.items()))
            raw = chat(args.judge, [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": prompt}], 2048, True)
            got = {k.lower(): v for k, v in parse_json_object(raw).items()}
            verdict = {o: (got.get(o.lower()) if isinstance(got.get(o.lower()), bool) else None)
                       for o in pairs}
            cache[key] = {"key": key, "query_id": row["query_id"], "verdict": verdict,
                          "raw": raw[-1500:]}
        return item, cache[key]["verdict"]

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(judge, items))
    args.cache.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n"
                                  for c in cache.values()))

    n_pairs = n_true = n_false = n_none = 0
    q_all_pass = q_judged = 0
    by_cell = {}
    for (cell, _, _, _), v in results:
        if not v:
            continue
        q_judged += 1
        vals = list(v.values())
        n_pairs += len(vals)
        n_true += sum(x is True for x in vals)
        n_false += sum(x is False for x in vals)
        n_none += sum(x is None for x in vals)
        q_all_pass += all(x is True for x in vals)
        c = by_cell.setdefault(cell, [0, 0])
        c[0] += len(vals)
        c[1] += sum(x is True for x in vals)

    summary = {
        "generator_model": GENERATOR,
        "judge_model": args.judge,
        "prompt_version": PROMPT_VERSION,
        "queries_aliased": len(items),
        "queries_judged": q_judged,
        "alias_pairs": n_pairs,
        "pairs_preserved": n_true,
        "pairs_changed": n_false,
        "pairs_unparsed": n_none,
        "pair_pass_rate": round(n_true / n_pairs, 4) if n_pairs else None,
        "query_all_pass_rate": round(q_all_pass / q_judged, 4) if q_judged else None,
        "pair_pass_rate_by_cell": {c: round(t / n, 4) for c, (n, t) in by_cell.items() if n},
        "queries_alias_complete": sum(a["alias_complete"] for _, _, _, a in items),
    }
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Human spot-check sample.
    rng = random.Random(args.seed)
    pool = [r for r in results if r[1]]
    sample = rng.sample(pool, min(args.n_sample, len(pool)))
    lines = [f"# Alias fidelity spot check ({len(sample)} queries, seed {args.seed})",
             "", f"Generator: `{GENERATOR}`. Judge: `{args.judge}`. "
             "Judge verdict per term in brackets.", ""]
    for (cell, repo, row, a), v in sample:
        text = row["problem_statement_stripped"]
        lines += [f"## {row['query_id']} (cell {cell})", "", "| original | alias | judge |",
                  "|---|---|---|"]
        for t, alias in a["aliases"].items():
            o = (surface_forms(text, t) or [t])[0]
            lines.append(f"| `{o}` | `{alias}` | {v.get(o, 'n/a') if alias else 'not aliased'} |")
        clip = lambda s: s[:1200].replace("```", "'''")  # noqa: E731
        lines += ["", "**Original (stripped):**", "", "```", clip(text), "```", "",
                  "**Alias:**", "", "```", clip(row["alias_query"]), "```", ""]
    args.out_md.write_text("\n".join(lines))
    print(f"wrote {args.out_json} and {args.out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
