#!/usr/bin/env python3
"""Non-lossy anchor ablation: sever the query->gold lexical bridge, hold the
rest of the query fixed.

Why this exists
---------------
`sec:genarm` shows that *adding* lexical anchors removes the dense premium.
The mirror experiment -- does *removing* anchors restore it? -- was attempted
with the cellC LLM rewrites and failed to be interpretable: those rewrites
compress a full PR description to one or two sentences, so BM25 lost 0.041
while dense lost 0.08-0.13, and the length/lossiness confound dominated the
anchor manipulation. The conclusion recorded at the time was that an
identifier-preserving *paraphrase* would not rescue the design either.

This script takes the other route: no paraphrase, no regeneration, no length
change. It edits only the tokens that form the lexical bridge, in place.

  ablated  the K rarest query tokens that also occur in the gold file are
           replaced by deterministic synthetic identifiers of the same
           length and case pattern, absent from the corpus. Every other
           character of the query is byte-identical to the original.
  control  the same number of query tokens, IDF-matched but NOT shared with
           the gold file, are replaced the same way. This arm perturbs just
           as many rare tokens without severing the bridge, so it separates
           "we damaged the query" from "we removed the anchors".

Prediction, if the premium is a property of the query's lexical anchors:
BM25 falls sharply on `ablated` and only slightly on `control`, dense is
close to flat on both (the prose it grounds on is untouched), so the premium
*rises* on `ablated` and not on `control`. Because dense-on-original is
already cached, the scoring arms reuse smoke_test_django.py exactly as the
generative arm does: --query-field ablated_query / control_query.

Emits augmented qrel files plus per-query accounting (which tokens went, their
IDF, tau before/after, char-length delta) so the manipulation can be audited
before any retrieval is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_retrieval as _std  # noqa: E402
from run_retrieval import (  # noqa: E402
    _CAMEL_BOUNDARY_RE,
    TOKEN_RE,
    chunk_file,
    compute_idf,
    list_corpus,
    read_file,
    strip_pr_boilerplate,
    tokenize,
)
from build_source_only_mirror import is_test  # noqa: E402
from recompute_tau_variants import tau_mean_topk, tau_top1  # noqa: E402

_VOWELS = "aeiou"
_CONSONANTS = "bcdfghjklmnpqrstvwxz"


def surrogate(token: str, corpus_vocab: set[str], salt: int = 0) -> str | None:
    """Deterministic pseudo-identifier of the same length, absent from corpus.

    Alternating consonant/vowel keeps it a single TOKEN_RE match (no digits or
    underscores introduced, so tokenization is unchanged) and keeps it
    pronounceable enough to read in a diff.

    Returns None when no same-length surrogate exists -- in practice only for
    one-character tokens, where the candidate space is 20 consonants and a
    large corpus already contains all of them. The caller then leaves the token
    alone. Padding to a different length is deliberately NOT done: exact length
    preservation is the property that separates this manipulation from the
    lossy cellC rewrite, and it is worth more than ablating one short token.
    """
    for attempt in range(64):
        h = hashlib.sha256(f"{token}|{salt}|{attempt}".encode()).digest()
        out = []
        for i in range(len(token)):
            pool = _CONSONANTS if i % 2 == 0 else _VOWELS
            out.append(pool[h[i % len(h)] % len(pool)])
        cand = "".join(out)
        if cand not in corpus_vocab and cand != token:
            return cand
    return None


def match_case(original: str, replacement: str) -> str:
    """Carry the original surface casing onto the replacement, char by char."""
    out = []
    for i, ch in enumerate(replacement):
        src = original[i] if i < len(original) else original[-1]
        out.append(ch.upper() if src.isupper() else ch)
    return "".join(out)


def segments(tok: str) -> list[str]:
    """Split an identifier into the units `tokenize` emits under camelCase
    mode, keeping the underscores as their own segments so that rejoining is
    byte-lossless: `get_UserById` -> [get, _, User, By, Id].

    Needed because the ablation maps *tokens*, and under
    `--tokenize-camelcase` a token is a subword of an identifier, not the
    whole TOKEN_RE match. Substituting at whole-identifier granularity while
    scoring at subword granularity would silently ablate nothing.
    """
    out: list[str] = []
    for i, part in enumerate(tok.split("_")):
        if i:
            out.append("_")
        if part:
            out.extend(s for s in _CAMEL_BOUNDARY_RE.split(part) if s)
    return out


def substitute(text: str, mapping: dict[str, str]) -> str:
    """Replace whole-token occurrences (case-insensitively) using `mapping`.

    Length is preserved exactly, because each surrogate is built to the length
    of the token it replaces. Under camelCase tokenization the replacement
    unit is the subword, matching what `tokenize` -- and therefore BM25 --
    actually indexes.
    """
    if not mapping:
        return text

    camel = _std.TOKENIZE_CAMELCASE

    def repl(m):
        tok = m.group(0)
        if not camel:
            sub = mapping.get(tok.lower())
            return match_case(tok, sub) if sub else tok
        out = []
        for seg in segments(tok):
            sub = None if seg == "_" else mapping.get(seg.lower())
            out.append(match_case(seg, sub) if sub else seg)
        return "".join(out)

    return TOKEN_RE.sub(repl, text)


def build_corpus_index(clone_dir: Path, exts: tuple[str, ...],
                       chunk_chars: int, overlap: int):
    paths = list_corpus(clone_dir, exts)
    texts = [read_file(clone_dir, p) for p in paths]
    chunk_texts: list[str] = []
    ranges: list[tuple[int, int]] = []
    for t in texts:
        cs = chunk_file(t, chunk_chars, overlap)
        start = len(chunk_texts)
        chunk_texts.extend(cs)
        ranges.append((start, start + len(cs)))
    chunk_tokens = [tokenize(c) for c in chunk_texts]
    idf = compute_idf(chunk_tokens)
    vocab = set()
    for ct in chunk_tokens:
        vocab.update(ct)
    return paths, ranges, chunk_tokens, idf, vocab


def gold_token_set(gold_file, paths, ranges, chunk_tokens) -> set[str]:
    if gold_file not in set(paths):
        return set()
    gidx = paths.index(gold_file)
    gs, ge = ranges[gidx]
    out: set[str] = set()
    for ci in range(gs, ge):
        out.update(chunk_tokens[ci])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qrel-file", type=Path, required=True)
    ap.add_argument("--clone-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, default=None,
                    help="If given, restrict to the query ids in this cell's "
                         "cached per-query file, so the arm is evaluated on "
                         "exactly the canonical query set.")
    ap.add_argument("--repo-name", default=None,
                    help="Subdirectory under --results-root (defaults to the "
                         "qrel file stem).")
    ap.add_argument("--strip-boilerplate", action="store_true")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Number of rarest shared tokens to ablate (default 5, "
                         "matching the top-5 definition of tau_mean5).")
    ap.add_argument("--code-exts", default=".py")
    ap.add_argument("--tokenize-camelcase", action="store_true",
                    help="Split camelCase identifiers into subwords, matching "
                         "the R9 cross-language BM25 configuration. Required "
                         "for TS/JS/Rust/Go so that the ablation is scored "
                         "against the same token space R9 indexed.")
    ap.add_argument("--chunk-chars", type=int, default=1000)
    ap.add_argument("--chunk-overlap-lines", type=int, default=10)
    args = ap.parse_args()

    # Must be set before any tokenize() call: `tokenize` reads this from its
    # own module globals at call time.
    if args.tokenize_camelcase:
        _std.TOKENIZE_CAMELCASE = True

    exts = tuple(e.strip() for e in args.code_exts.split(",") if e.strip())
    rows = [json.loads(l) for l in args.qrel_file.read_text().splitlines() if l.strip()]

    if args.results_root:
        repo = args.repo_name or args.qrel_file.stem
        cached = args.results_root / repo / "smoke_test_django.jsonl"
        wanted = [json.loads(l)["query_id"]
                  for l in cached.read_text().splitlines() if l.strip()]
        by_id = {r["query_id"]: r for r in rows}
        rows = [by_id[q] for q in wanted if q in by_id]

    print(f"[load] {len(rows)} queries from {args.qrel_file.name}", file=sys.stderr)
    paths, ranges, chunk_tokens, idf, vocab = build_corpus_index(
        args.clone_dir, exts, args.chunk_chars, args.chunk_overlap_lines)
    print(f"[corpus] {len(paths)} files, {len(chunk_tokens)} chunks, "
          f"{len(vocab)} vocab", file=sys.stderr)

    out_rows, audit = [], []
    for row in rows:
        text = row.get("problem_statement") or ""
        if args.strip_boilerplate:
            text = strip_pr_boilerplate(text)
        q_toks = tokenize(text)
        q_set = set(q_toks)
        # Anchor tokens come from the first gold file that is indexed and not a test (as for tau).
        # gold_files[0] alone is often a changelog or doc file on raw PRs; it gave an empty gold
        # set, so both arms came out byte-identical to the original query.
        path_set = set(paths)
        first_gold = next((g for g in row["gold_files"] if g in path_set and not is_test(g)), None)
        gold = gold_token_set(first_gold, paths, ranges, chunk_tokens) if first_gold else set()
        shared = q_set & gold

        # --- ablation arm: the K rarest shared tokens (the lexical bridge)
        # Ties in IDF are broken alphabetically: set order varies with Python's hash seed.
        ranked = sorted(shared, key=lambda t: (-idf.get(t, 0.0), t))[:args.top_k]
        abl_map, unablatable = {}, []
        for t in ranked:
            sur = surrogate(t, vocab, salt=0)
            if sur is None:
                unablatable.append(t)
            else:
                abl_map[t] = sur

        # --- control arm: K unshared query tokens, IDF-matched to the ablated
        unshared = sorted(q_set - shared, key=lambda t: (-idf.get(t, 0.0), t))
        ctrl_tokens = []
        used = set()
        for t in ranked:
            target = idf.get(t, 0.0)
            best = min((c for c in unshared if c not in used),
                       key=lambda c: (abs(idf.get(c, 0.0) - target), c), default=None)
            if best is not None:
                ctrl_tokens.append(best)
                used.add(best)
        ctrl_map = {t: sur for t in ctrl_tokens
                    if (sur := surrogate(t, vocab, salt=1)) is not None}

        ablated = substitute(text, abl_map)
        control = substitute(text, ctrl_map)

        abl_shared = set(tokenize(ablated)) & gold
        ctrl_shared = set(tokenize(control)) & gold

        new_row = dict(row)
        new_row["problem_statement_stripped"] = text
        new_row["ablated_query"] = ablated
        new_row["control_query"] = control
        out_rows.append(new_row)

        audit.append({
            "query_id": row["query_id"],
            "anchor_gold_file": first_gold,
            "n_shared_before": len(shared),
            "n_shared_after_ablation": len(abl_shared),
            "n_shared_after_control": len(ctrl_shared),
            "n_ablated": len(abl_map),
            # True when we mapped tokens but the query->gold bridge did not
            # shrink -- i.e. the substitution did not land. The camelCase
            # granularity mismatch is the way this happens silently.
            "ablation_noop": bool(abl_map) and abl_shared == shared,
            "n_control_substituted": len(ctrl_map),
            "ablated_tokens": [[t, round(idf.get(t, 0.0), 4)] for t in ranked],
            "unablatable_tokens": unablatable,
            "control_tokens": [[t, round(idf.get(t, 0.0), 4)] for t in ctrl_tokens],
            "tau_mean5_before": tau_mean_topk(shared, idf, 5),
            "tau_mean5_ablated": tau_mean_topk(abl_shared, idf, 5),
            "tau_mean5_control": tau_mean_topk(ctrl_shared, idf, 5),
            "tau_top1_before": tau_top1(shared, idf),
            "tau_top1_ablated": tau_top1(abl_shared, idf),
            "tau_top1_control": tau_top1(ctrl_shared, idf),
            "chars_original": len(text),
            "chars_ablated": len(ablated),
            "chars_control": len(control),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.qrel_file.stem
    (args.out_dir / f"{stem}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
    (args.out_dir / f"{stem}_audit.jsonl").write_text(
        "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audit))

    n = len(audit) or 1
    dl = sum(a["chars_ablated"] - a["chars_original"] for a in audit) / n
    print(json.dumps({
        "queries": len(audit),
        "mean_chars_original": round(sum(a["chars_original"] for a in audit) / n, 1),
        "mean_char_delta_ablated": round(dl, 4),
        "mean_shared_before": round(sum(a["n_shared_before"] for a in audit) / n, 2),
        "mean_shared_after_ablation": round(sum(a["n_shared_after_ablation"] for a in audit) / n, 2),
        "mean_shared_after_control": round(sum(a["n_shared_after_control"] for a in audit) / n, 2),
        "mean_tau_mean5_before": round(sum(a["tau_mean5_before"] for a in audit) / n, 4),
        "mean_tau_mean5_ablated": round(sum(a["tau_mean5_ablated"] for a in audit) / n, 4),
        "mean_tau_mean5_control": round(sum(a["tau_mean5_control"] for a in audit) / n, 4),
        "queries_with_length_change": sum(
            a["chars_ablated"] != a["chars_original"] for a in audit),
        "queries_with_unablatable_token": sum(
            bool(a["unablatable_tokens"]) for a in audit),
        "queries_ablation_noop": sum(bool(a["ablation_noop"]) for a in audit),
        "tokenize_camelcase": _std.TOKENIZE_CAMELCASE,
    }, indent=2))

    # A no-op ablation still produces a complete, plausible-looking output
    # file, so refuse to exit clean on one.
    n_noop = sum(bool(a["ablation_noop"]) for a in audit)
    if n_noop > 0.1 * n:
        print(f"[FATAL] ablation did not land on {n_noop}/{len(audit)} queries: "
              f"tokens were mapped but the query->gold overlap is unchanged. "
              f"With --tokenize-camelcase this usually means the substitution "
              f"granularity does not match the tokenizer.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
