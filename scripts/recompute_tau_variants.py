"""R3 robustness — recompute alternative τ formulations per query.

Outputs four τ variants alongside the existing baseline so we can pick the
formulation with the tightest correlation to Δ (= nDCG_dense − nDCG_BM25).

All variants are set/IDF-based (no BM25 inside the script) so the recompute is
CPU-cheap (seconds per repo, vs minutes for any BM25-dependent path).

Variants:
  - tau_mean5    : mean IDF of top-5 rarest shared tokens (existing baseline)
  - tau_top1     : IDF of the single rarest shared token
                   (smoother at the floor — does it still collapse to 0 when
                   query has zero shared tokens?)
  - tau_mean_all : mean IDF over ALL shared tokens (no top-K truncation)
                   (more stable; less sensitive to query length)
  - tau_jaccard  : weighted Jaccard between query tokens and full gold-file
                   token set, where weights are IDF (rewards rare-token overlap)

This script is CPU-only; no dense model is loaded. It reuses helpers from
smoke_test_django.py without modifying that file.

Usage (mirrors smoke_test_django CLI):
  python recompute_tau_variants.py \
      --qrel-file ${QRELDIR}/${REPO}.jsonl \
      --clone-dir ${CLONEDIR}/${REPO} \
      --out-dir   ${OUT_DIR} \
      --strip-boilerplate \
      --chunk-chars 1000 \
      --n 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Reuse helpers from the existing smoke-test script without modifying it.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from build_source_only_mirror import is_test  # noqa: E402
from run_retrieval import (  # noqa: E402
    chunk_file,
    compute_idf,
    list_corpus,
    load_qrels,
    read_file,
    strip_pr_boilerplate,
    tokenize,
)


def tau_mean_topk(shared: set[str], idf: dict[str, float], top_k: int) -> float:
    if not shared:
        return 0.0
    ranked = sorted(shared, key=lambda t: -idf.get(t, 0.0))[:top_k]
    vals = [idf.get(t, 0.0) for t in ranked]
    return float(np.mean(vals)) if vals else 0.0


def tau_top1(shared: set[str], idf: dict[str, float]) -> float:
    if not shared:
        return 0.0
    return float(max(idf.get(t, 0.0) for t in shared))


def tau_mean_all(shared: set[str], idf: dict[str, float]) -> float:
    if not shared:
        return 0.0
    vals = [idf.get(t, 0.0) for t in shared]
    return float(np.mean(vals)) if vals else 0.0


def tau_jaccard_idf(q_set: set[str], gold_set: set[str],
                    idf: dict[str, float]) -> float:
    """IDF-weighted Jaccard: sum(IDF of intersection) / sum(IDF of union)."""
    if not q_set or not gold_set:
        return 0.0
    inter = q_set & gold_set
    union = q_set | gold_set
    num = sum(idf.get(t, 0.0) for t in inter)
    den = sum(idf.get(t, 0.0) for t in union)
    return float(num / den) if den > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrel-file", type=Path, required=True)
    ap.add_argument("--clone-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--strip-boilerplate", action="store_true", default=True,
                    help="Strip PR-template boilerplate (default: on, as in the paper)")
    ap.add_argument("--no-strip-boilerplate", dest="strip_boilerplate", action="store_false")
    ap.add_argument("--chunk-chars", type=int, default=1000)
    ap.add_argument("--chunk-overlap-lines", type=int, default=10)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--code-exts", default=".py",
                    help="Comma-separated corpus file extensions (match the smoke run)")
    ap.add_argument("--tokenize-camelcase", action="store_true",
                    help="Split camelCase identifiers (match the smoke run)")
    args = ap.parse_args()

    if args.tokenize_camelcase:
        import run_retrieval as _std
        _std.TOKENIZE_CAMELCASE = True
    exts = tuple(e.strip() for e in args.code_exts.split(",") if e.strip())

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] qrels from {args.qrel_file}", file=sys.stderr)
    qrels = load_qrels(args.qrel_file, args.n, args.seed)
    print(f"[load] {len(qrels)} queries", file=sys.stderr)

    if args.strip_boilerplate:
        for q in qrels:
            q["problem_statement"] = strip_pr_boilerplate(q["problem_statement"])

    paths = list_corpus(args.clone_dir, exts)
    print(f"[corpus] {len(paths)} files", file=sys.stderr)
    texts = [read_file(args.clone_dir, p) for p in paths]
    chunk_texts: list[str] = []
    file_chunk_ranges: list[tuple[int, int]] = []
    for t in texts:
        cs = chunk_file(t, args.chunk_chars, args.chunk_overlap_lines)
        start = len(chunk_texts)
        chunk_texts.extend(cs)
        file_chunk_ranges.append((start, start + len(cs)))
    print(f"[corpus] {len(chunk_texts)} chunks", file=sys.stderr)
    chunk_tokens = [tokenize(c) for c in chunk_texts]

    print(f"[idf] computing chunk-level IDF...", file=sys.stderr)
    idf = compute_idf(chunk_tokens)

    path_set = set(paths)
    out_path = args.out_dir / "tau_variants.jsonl"
    n_zero_top1 = 0
    n_zero_mean5 = 0
    n_zero_jacc = 0
    with out_path.open("w") as fh:
        for q in qrels:
            qid = q["query_id"]
            query = q["problem_statement"]
            q_toks = tokenize(query)
            q_set = set(q_toks)

            # tau is measured against the first gold file that is indexed and not a test (the
            # indexed labels); gold_files[0] alone is often a changelog or doc file on raw PRs,
            # which the corpus does not index and which silently gave tau = 0.
            first_gold = next((g for g in q["gold_files"] if g in path_set and not is_test(g)), None)
            if first_gold is not None:
                gidx = paths.index(first_gold)
                gs, ge = file_chunk_ranges[gidx]
                gold_set: set[str] = set()
                for ci in range(gs, ge):
                    gold_set.update(chunk_tokens[ci])
            else:
                gold_set = set()

            shared = q_set & gold_set
            t_mean5 = tau_mean_topk(shared, idf, top_k=5)
            t_top1 = tau_top1(shared, idf)
            t_mean_all = tau_mean_all(shared, idf)
            t_jacc = tau_jaccard_idf(q_set, gold_set, idf)

            n_zero_mean5 += int(t_mean5 == 0.0)
            n_zero_top1 += int(t_top1 == 0.0)
            n_zero_jacc += int(t_jacc == 0.0)

            fh.write(json.dumps({
                "query_id": qid,
                "tau_mean5": t_mean5,
                "tau_top1": t_top1,
                "tau_mean_all": t_mean_all,
                "tau_jaccard": t_jacc,
                "n_q_tokens": len(q_set),
                "n_gold_tokens": len(gold_set),
                "n_shared": len(shared),
            }) + "\n")

    print(f"[done] wrote {out_path}", file=sys.stderr)
    print(f"[zero-floor] mean5={n_zero_mean5}/{len(qrels)}  "
          f"top1={n_zero_top1}/{len(qrels)}  "
          f"jacc={n_zero_jacc}/{len(qrels)}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
