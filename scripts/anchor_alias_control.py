#!/usr/bin/env python3
"""Alias control for the anchor ablation (review finding #14).

Why this exists
---------------
`anchor_ablation.py` replaces the K rarest query tokens shared with the gold
file by same-length NONSENSE strings. Both retrievers collapse. A reviewer can
object that nonsense destroys task meaning as well as the lexical bridge: an
encoder that keys on what `parse_header` *means* -- without literal matching --
would also collapse, so the ablation cannot tell lexical dependence from
semantic damage.

This arm removes the bridge but keeps the meaning. Each of the SAME tokens the
ablated arm replaced is renamed to a meaning-preserving alias -- a different
identifier (or word) a developer would read as the same concept -- that occurs
nowhere in the repository corpus. BM25 loses the match exactly as in the
ablated arm (the alias is out of vocabulary); an encoder doing semantic
matching should not.

  alias ~ ablated  (dense)  -> the encoder needs the literal token: lexical
  alias ~ guard    (dense)  -> the encoder needs the meaning: ablation was
                               semantic damage, #14 stands

Inputs are the existing ablation outputs (`qrels_cell{A,B}_ablated/*.jsonl`
plus `*_audit.jsonl`), so the query set, the stripped text and the selected
tokens are reused byte-for-byte; the nonsense (`ablated_query`) and placebo
(`control_query`) fields are carried through untouched. Output rows add
`alias_query`, scored by the existing slurm scripts with
`--query-field alias_query`.

Aliases come from an LLM through the OpenAI-compatible proxy at temperature 0
with a fixed prompt, cached per query so a rerun is deterministic and
resumable. Every alias is verified:

  * a single TOKEN_RE identifier (so token count is unchanged),
  * absent from the corpus vocabulary (so BM25 cannot match it anywhere),
  * not a casing/underscore/word-order variant of the original and not
    containing it (the camelCase no-op trap, in a new form),
  * after substitution the query->gold shared set must equal the ablated
    arm's, i.e. the bridge is severed exactly as much.

Rejected aliases are re-requested with the reasons; a query whose tokens
cannot all be aliased is kept but flagged `alias_complete=false`, so the
analysis can be run on all queries and on the complete subset.

Also writes the review-#14 manipulation audit across all three arms: token
types and token OCCURRENCES replaced per query, IDF match between the ablated
and placebo tokens, and how many replaced tokens are identifiers versus plain
words.

Python cells only: under --tokenize-camelcase a token is a subword and the
whole-identifier alias substitution below would not land (see
coreb_r9_anchor_ablation trap 1), so camel mode is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anchor_ablation import build_corpus_index, gold_token_set  # noqa: E402
from run_retrieval import TOKEN_RE, tokenize  # noqa: E402

PROMPT_VERSION = "alias-v1"

SYSTEM = (
    "You rename words in software issue reports. You answer with a single "
    "JSON object and nothing else."
)

USER_TMPL = """Below is an issue report from the {repo} repository, followed by a list of terms that appear in it.

For EACH term, propose a replacement that a developer reading the report would understand as meaning exactly the same thing -- the way a careful refactoring renames an identifier, or the way a synonym replaces a word.

Rules for every replacement:
1. It must be ONE token made only of letters, digits and underscores (no spaces, dots or hyphens). To express a phrase, join words with underscores, e.g. "because" -> "for_the_reason_that".
2. Keep the term's naming style: snake_case stays snake_case, CamelCase stays CamelCase, UPPER_CASE stays UPPER_CASE, a plain lowercase word stays lowercase.
3. It must use DIFFERENT words from the original. Changing only capitalisation or underscores, reordering the same words, or adding a prefix/suffix to the original is NOT allowed. It must not contain the original term.
4. Prefer an uncommon but natural wording: the replacement must not be a name that already exists in the {repo} codebase.
5. Preserve the meaning in the context of this report as closely as possible.
{feedback}
Issue report:
<<<
{text}
>>>

Terms (as they appear in the report):
{terms}

Answer with a JSON object mapping each term exactly as listed to its replacement, e.g. {{"parse_header": "read_header_fields", "because": "for_the_reason_that"}}."""

FEEDBACK_TMPL = """
Your previous replacements for some terms were rejected:
{reasons}
Give NEW replacements for those terms that satisfy every rule. Common single-word synonyms
usually already exist in the codebase, so prefer a phrase joined with underscores
(e.g. "gets" -> "obtains_hold_of", "because" -> "for_the_reason_that").
"""

_WORD_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|_")


def subwords(tok: str) -> list[str]:
    return [s.lower() for s in _WORD_SPLIT.split(tok) if s]


def is_identifier_like(surface: str) -> bool:
    """Heuristic for the audit: does this token look like code rather than prose?"""
    return ("_" in surface or any(c.isdigit() for c in surface)
            or (any(c.isupper() for c in surface[1:]))
            or (surface.isupper() and len(surface) > 1))


def surface_forms(text: str, token: str) -> list[str]:
    return [m.group(0) for m in TOKEN_RE.finditer(text) if m.group(0).lower() == token]


def reject_reason(orig_surface: str, alias: str, vocab: set[str]) -> str | None:
    if not isinstance(alias, str) or not TOKEN_RE.fullmatch(alias or ""):
        return "not a single letters/digits/underscores token"
    a, o = alias.lower(), orig_surface.lower()
    if a == o:
        return "identical to the original"
    if a.replace("_", "") == o.replace("_", ""):
        return "only changes underscores or capitalisation"
    if sorted(subwords(alias)) == sorted(subwords(orig_surface)):
        return "only reorders the original's words"
    if len(o) >= 3 and o in a:
        return "contains the original term"
    if set(tokenize(alias)) & vocab:
        return "already occurs in the codebase"
    return None


def apply_case(surface: str, alias: str) -> str:
    """Carry the occurrence's surface style onto the alias without assuming
    equal length (unlike anchor_ablation.match_case)."""
    if surface.isupper() and len(surface) > 1:
        return alias.upper()
    if surface.islower():
        return alias.lower()
    if surface[:1].isupper() and surface[1:].islower() and alias[:1].islower():
        return alias[:1].upper() + alias[1:]
    return alias


def substitute_alias(text: str, mapping: dict[str, str]) -> str:
    def repl(m):
        tok = m.group(0)
        sub = mapping.get(tok.lower())
        return apply_case(tok, sub) if sub else tok
    return TOKEN_RE.sub(repl, text)


def chat(model: str, messages: list[dict], max_tokens: int, think: bool) -> str:
    base = os.environ.get("OPENAI_API_BASE")
    key = os.environ.get("OPENAI_API_KEY")
    if not base or not key:
        raise SystemExit("set OPENAI_API_BASE and OPENAI_API_KEY "
                         "in the environment; never pass the key on argv")
    url = base.rstrip("/") + ("/chat/completions" if base.rstrip("/").endswith("/v1")
                              else "/v1/chat/completions")
    body = json.dumps({"model": model, "messages": messages, "temperature": 0,
                       "seed": 0, "max_tokens": max_tokens,
                       # Thinking runs out the budget mid-reasoning (empty content)
                       # and is less reproducible at temperature 0.
                       **({} if think else
                          {"chat_template_kwargs": {"enable_thinking": False}})}).encode()
    for attempt in range(5):
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            # Cloudflare in front of the proxy rejects urllib's default UA (1010).
            "User-Agent": "coreb-alias-control/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.loads(r.read())
            content = out["choices"][0]["message"].get("content") or ""
            if content.strip():
                return content
            # Empty content: budget exhausted. Retry rather than burn a round.
            print(f"[empty content, attempt {attempt + 1}]", file=sys.stderr)
            continue
        except (urllib.error.URLError, TimeoutError, KeyError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 403):
                raise
            if attempt == 4:
                raise
            print(f"[retry {attempt + 1}] {e}", file=sys.stderr)
            time.sleep(10 * (attempt + 1))
    return ""


def parse_json_object(s: str) -> dict:
    # Thinking models may leave reasoning in content; take the last {...}.
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", s, flags=re.S))):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {}


def alias_query(row, tokens, repo, vocab, args, cache: dict) -> dict:
    """Return {token: alias or None} after up to --max-rounds verified rounds."""
    text = row["problem_statement_stripped"]
    surf = {t: (surface_forms(text, t) or [t])[0] for t in tokens}
    key = hashlib.sha256(json.dumps(
        [PROMPT_VERSION, args.model, row["query_id"], sorted(tokens)]).encode()).hexdigest()
    if key in cache:
        return cache[key]["mapping"]

    final: dict[str, str | None] = {t: None for t in tokens}
    pending = list(tokens)
    rejected: dict[str, str] = {}
    rounds = []
    for rnd in range(args.max_rounds):
        if not pending:
            break
        feedback = ""
        if rejected:
            feedback = FEEDBACK_TMPL.format(reasons="\n".join(
                f'- "{surf[t]}" -> "{a}": {why}' for t, (a, why) in rejected.items()))
        prompt = USER_TMPL.format(
            repo=repo, text=text[:args.max_query_chars], feedback=feedback,
            terms="\n".join(f"- {surf[t]}" for t in pending))
        raw = chat(args.model, [{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": prompt}], args.max_tokens, args.think)
        got = {k.lower(): v for k, v in parse_json_object(raw).items()}
        rejected = {}
        for t in pending:
            a = got.get(surf[t].lower()) or got.get(t)
            why = reject_reason(surf[t], a, vocab) if a is not None else "missing"
            # Two terms must not share an alias: that would merge two concepts.
            if why is None and a.lower() in {v.lower() for v in final.values() if v}:
                why = "duplicates another replacement"
            if why is None:
                final[t] = a
            else:
                rejected[t] = (a, why)
        rounds.append({"pending": pending, "rejected": rejected, "raw": raw[-2000:]})
        pending = list(rejected)

    cache[key] = {"query_id": row["query_id"], "mapping": final, "rounds": rounds,
                  "model": args.model, "prompt_version": PROMPT_VERSION}
    return final


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablated-qrel", type=Path, required=True,
                    help="qrels_cell{A,B}_ablated/{repo}.jsonl from anchor_ablation.py")
    ap.add_argument("--clone-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="qwen35-122B")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--think", action="store_true",
                    help="Leave the model's reasoning mode on (default off).")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="Completion budget; the thinking model spends most of it reasoning.")
    ap.add_argument("--max-query-chars", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="dry run on the first N queries")
    ap.add_argument("--code-exts", default=".py")
    ap.add_argument("--chunk-chars", type=int, default=1000)
    ap.add_argument("--chunk-overlap-lines", type=int, default=10)
    ap.add_argument("--tokenize-camelcase", action="store_true")
    args = ap.parse_args()

    if args.tokenize_camelcase:
        print("[FATAL] camelCase mode is not supported: whole-identifier aliases "
              "would not land on subword tokens (r9 trap 1).", file=sys.stderr)
        return 2

    stem = args.ablated_qrel.stem
    rows = [json.loads(l) for l in args.ablated_qrel.read_text().splitlines() if l.strip()]
    audit_in = {a["query_id"]: a for a in map(json.loads, (
        args.ablated_qrel.parent / f"{stem}_audit.jsonl").read_text().splitlines())}
    if args.limit:
        rows = rows[:args.limit]

    exts = tuple(e.strip() for e in args.code_exts.split(",") if e.strip())
    paths, ranges, chunk_tokens, idf, vocab = build_corpus_index(
        args.clone_dir, exts, args.chunk_chars, args.chunk_overlap_lines)
    print(f"[corpus] {stem}: {len(paths)} files, {len(vocab)} vocab; "
          f"{len(rows)} queries", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / f"{stem}_llm_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for l in cache_path.read_text().splitlines():
            c = json.loads(l)
            cache[c["key"]] = c

    def work(row):
        a = audit_in[row["query_id"]]
        # Exactly the tokens the nonsense arm actually replaced.
        tokens = [t for t, _ in a["ablated_tokens"] if t not in set(a["unablatable_tokens"])]
        return row, a, tokens, alias_query(row, tokens, stem, vocab, args, cache)

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(work, rows))

    with cache_path.open("w") as f:
        for k, c in cache.items():
            f.write(json.dumps({"key": k, **c}, ensure_ascii=False) + "\n")

    out_rows, audit = [], []
    for row, a, tokens, mapping in results:
        text = row["problem_statement_stripped"]
        good = {t: v for t, v in mapping.items() if v}
        aliased = substitute_alias(text, good)
        gold = gold_token_set(row["gold_files"][0], paths, ranges, chunk_tokens)
        shared = set(tokenize(text)) & gold
        shared_alias = set(tokenize(aliased)) & gold
        shared_abl = set(tokenize(row["ablated_query"])) & gold

        q_toks = tokenize(text)
        ctrl = [t for t, _ in a["control_tokens"]]
        new_row = dict(row)
        new_row["alias_query"] = aliased
        out_rows.append(new_row)
        audit.append({
            "query_id": row["query_id"],
            "n_tokens_selected": len(tokens),
            "n_aliased": len(good),
            "alias_complete": len(good) == len(tokens),
            "aliases": {t: mapping[t] for t in tokens},
            # Aliases that keep a word of the original (to_native_string ->
            # convert_to_native_text): meaning-preserving by design, but a
            # subword tokenizer still sees partial lexical overlap.
            "n_alias_subword_overlap": sum(
                bool(set(subwords(v)) & set(subwords((surface_forms(text, t) or [t])[0])))
                for t, v in good.items()),
            # Bridge parity with the nonsense arm: the whole point of the arm.
            "bridge_matches_ablated": shared_alias == shared_abl,
            "n_shared_before": len(shared),
            "n_shared_after_alias": len(shared_alias),
            "n_shared_after_ablation": len(shared_abl),
            # Review #14 audit: types vs occurrences, IDF parity, identifier share.
            "occ_ablated": sum(t in set(tokens) for t in q_toks),
            "occ_control": sum(t in set(ctrl) for t in q_toks),
            "n_control_types": len(ctrl),
            "idf_ablated_mean": (sum(i for _, i in a["ablated_tokens"]) / len(a["ablated_tokens"])
                                 if a["ablated_tokens"] else None),
            "idf_control_mean": (sum(i for _, i in a["control_tokens"]) / len(a["control_tokens"])
                                 if a["control_tokens"] else None),
            "idf_abs_gap_mean": (sum(abs(x[1] - y[1]) for x, y in
                                     zip(a["ablated_tokens"], a["control_tokens"]))
                                 / max(1, min(len(a["ablated_tokens"]), len(a["control_tokens"])))),
            "n_identifier_like": sum(is_identifier_like((surface_forms(text, t) or [t])[0])
                                     for t in tokens),
            "chars_original": len(text),
            "chars_alias": len(aliased),
        })

    (args.out_dir / f"{stem}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
    (args.out_dir / f"{stem}_audit.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in audit))

    n = len(audit) or 1
    mean = lambda k: round(sum(x[k] for x in audit if x[k] is not None) / n, 4)  # noqa: E731
    summary = {
        "repo": stem, "queries": len(audit), "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "alias_complete": sum(x["alias_complete"] for x in audit),
        "bridge_matches_ablated": sum(x["bridge_matches_ablated"] for x in audit),
        "mean_tokens_selected": mean("n_tokens_selected"),
        "queries_lt_5_tokens": sum(x["n_tokens_selected"] < 5 for x in audit),
        "mean_occ_ablated": mean("occ_ablated"),
        "mean_occ_control": mean("occ_control"),
        "mean_idf_ablated": mean("idf_ablated_mean"),
        "mean_idf_control": mean("idf_control_mean"),
        "mean_idf_abs_gap": mean("idf_abs_gap_mean"),
        "alias_subword_overlap_share": round(
            sum(x["n_alias_subword_overlap"] for x in audit)
            / max(1, sum(x["n_aliased"] for x in audit)), 4),
        "identifier_like_share": round(
            sum(x["n_identifier_like"] for x in audit)
            / max(1, sum(x["n_tokens_selected"] for x in audit)), 4),
        "mean_char_delta_alias": round(
            sum(x["chars_alias"] - x["chars_original"] for x in audit) / n, 1),
    }
    print(json.dumps(summary, indent=2))
    (args.out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))

    # A complete-looking file with the bridge intact would fake the result.
    bad = sum(x["alias_complete"] and not x["bridge_matches_ablated"] for x in audit)
    if bad:
        print(f"[FATAL] {bad} fully-aliased queries keep a different query->gold "
              f"bridge than the nonsense arm", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
