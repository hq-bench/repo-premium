#!/usr/bin/env python3
"""Per-query ambiguity of the tau anchor: is the rarest shared token unique?

tau(q) is MaxIDF over tokens the query shares with its gold file, so it scores
a token's RARITY. Rarity is not uniqueness: a token can be rare corpus-wide and
still be defined in many files (overrides across subclasses, vendored copies,
a test mirroring an implementation name). An internal reviewer raised exactly
this -- that a search tool hands the caller several same-named candidates and
leaves disambiguation to the LLM -- so this script measures how often the
anchor tau keys on is ambiguous, and lets the tau-effect be re-read within
strata of that ambiguity.

Emits, per query:
  t_star       the argmax-IDF shared token, i.e. the token tau_top1 scores
  tau_top1     its IDF (recomputed here, to stay self-contained)
  file_df      number of distinct corpus FILES containing t_star
  gold_share   |gold_files containing t_star| / file_df
  n_shared     number of shared query/gold tokens

Chunking, tokenizer and IDF come from smoke_test_django, so the numbers line
up with the canonical pipeline. Pass the same --chunk-chars / --code-exts /
--tokenize-camelcase the scored run used.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from build_source_only_mirror import is_test  # noqa: E402
from run_retrieval import (  # noqa: E402
    chunk_file, compute_idf, list_corpus, load_qrels, read_file,
    strip_pr_boilerplate, tokenize,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qrel-file", type=Path, required=True)
    ap.add_argument("--clone-dir", type=Path, required=True)
    ap.add_argument("--out-file", type=Path, required=True)
    ap.add_argument("--strip-boilerplate", action="store_true", default=True,
                    help="Strip PR-template boilerplate (default: on, as in the paper)")
    ap.add_argument("--no-strip-boilerplate", dest="strip_boilerplate", action="store_false")
    ap.add_argument("--chunk-chars", type=int, default=1000)
    ap.add_argument("--chunk-overlap-lines", type=int, default=10)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--code-exts", default=".py")
    ap.add_argument("--tokenize-camelcase", action="store_true")
    args = ap.parse_args()

    if args.tokenize_camelcase:
        import run_retrieval as _std
        _std.TOKENIZE_CAMELCASE = True
    exts = tuple(e.strip() for e in args.code_exts.split(",") if e.strip())

    qrels = load_qrels(args.qrel_file, args.n, args.seed)
    if args.strip_boilerplate:
        for q in qrels:
            q["problem_statement"] = strip_pr_boilerplate(q["problem_statement"])

    paths = list_corpus(args.clone_dir, exts)
    texts = [read_file(args.clone_dir, p) for p in paths]
    chunk_texts: list[str] = []
    file_chunk_ranges: list[tuple[int, int]] = []
    for t in texts:
        cs = chunk_file(t, args.chunk_chars, args.chunk_overlap_lines)
        start = len(chunk_texts)
        chunk_texts.extend(cs)
        file_chunk_ranges.append((start, start + len(cs)))
    chunk_tokens = [tokenize(c) for c in chunk_texts]
    idf = compute_idf(chunk_tokens)

    # file-level token sets, and the inverted index token -> files
    file_tokens: list[set[str]] = []
    token_files: dict[str, set[int]] = defaultdict(set)
    for fi, (s, e) in enumerate(file_chunk_ranges):
        toks: set[str] = set()
        for ci in range(s, e):
            toks.update(chunk_tokens[ci])
        file_tokens.append(toks)
        for tok in toks:
            token_files[tok].add(fi)

    path_index = {p: i for i, p in enumerate(paths)}
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out_file.open("w") as fh:
        for q in qrels:
            q_set = set(tokenize(q["problem_statement"]))
            # Source-file gold, and the first indexed non-test gold file as tau's anchor file
            # (gold_files[0] alone is often an unindexed changelog on raw PRs, giving tau = 0).
            gold_paths = [g for g in q["gold_files"] if g in path_index and not is_test(g)]
            first_gold = gold_paths[0] if gold_paths else None
            gold_set = file_tokens[path_index[first_gold]] if first_gold else set()
            shared = q_set & gold_set
            if shared:
                t_star = max(shared, key=lambda t: (idf.get(t, 0.0), t))
                tau = float(idf.get(t_star, 0.0))
                files_with = token_files.get(t_star, set())
                file_df = len(files_with)
                gold_idx = {path_index[g] for g in gold_paths}
                gold_share = (len(files_with & gold_idx) / file_df) if file_df else 0.0
            else:
                t_star, tau, file_df, gold_share = "", 0.0, 0, 0.0
            fh.write(json.dumps({
                "query_id": q["query_id"],
                "t_star": t_star,
                "tau_top1": tau,
                "file_df": file_df,
                "gold_share": round(gold_share, 4),
                "n_shared": len(shared),
                "n_corpus_files": len(paths),
            }) + "\n")
            n += 1
    print(f"wrote {args.out_file}: {n} queries over {len(paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
