#!/usr/bin/env python3
"""Recompute tau_top1 (MaxIDF) and tau_mean5 for DeepSWE tasks.

Mirrors deepswe/retrieval.slurm: smoke_test_django helpers (server copy),
list_corpus over the task's repo_dir with CODE_EXTS, chunk 1000 / 10-line
overlap, chunk-level IDF, camelCase tokenization for typescript/javascript,
no boilerplate stripping. The analysis checks n_files against the stored
n_corpus_files and tau_mean5 against the stored tau_q before using tau_top1.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

SMOKE = None


def _init(smoke_dir: str) -> None:
    global SMOKE
    sys.path.insert(0, smoke_dir)
    import run_retrieval as s  # noqa: E402
    SMOKE = s


def one_task(row: dict) -> list[dict]:
    s = SMOKE
    s.TOKENIZE_CAMELCASE = row["language"] in ("typescript", "javascript")
    clone = Path(row["repo_dir"])
    exts = tuple(e.strip() for e in row["code_exts"].split(",") if e.strip())
    paths = s.list_corpus(clone, exts)
    chunk_tokens, ranges = [], {}
    for p in paths:
        cs = s.chunk_file(s.read_file(clone, p), 1000, 10)
        st = len(chunk_tokens)
        chunk_tokens.extend(s.tokenize(c) for c in cs)
        ranges[p] = (st, len(chunk_tokens))
    idf = s.compute_idf(chunk_tokens)
    out = []
    for l in Path(row["qrel_file"]).read_text().splitlines():
        if not l.strip():
            continue
        q = json.loads(l)
        q_toks = s.tokenize(q["problem_statement"])
        g = q["gold_files"][0]
        if g in ranges:
            a, b = ranges[g]
            gt = [t for ci in range(a, b) for t in chunk_tokens[ci]]
            shared = set(q_toks) & set(gt)
            mean5 = s.tau_q(q_toks, gt, idf, top_k=5)
            top1 = max((idf.get(t, 0.0) for t in shared), default=0.0)
        else:
            mean5 = top1 = 0.0
        out.append({"query_id": q["query_id"], "repo": q.get("repo"),
                    "n_files": len(paths), "tau_mean5": mean5, "tau_top1": top1})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--smoke", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    with open(a.manifest) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    with Pool(a.workers, initializer=_init, initargs=(a.smoke,)) as pool, \
            open(a.out, "w") as fo:
        for recs in pool.imap_unordered(one_task, rows):
            for r in recs:
                fo.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
