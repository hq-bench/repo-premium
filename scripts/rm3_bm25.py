#!/usr/bin/env python3
"""RM3 pseudo-relevance feedback on whole-file BM25 — the classical lexical
control for generate-then-grep.

Generate-then-grep adds model-written anchors to the query before BM25. RM3
adds terms mined from BM25's own top documents instead, with no model. If RM3
closes the premium as well, generation is just query expansion; if not, the
generator supplies evidence the corpus statistics cannot.

Setup (no tuning on the evaluation queries; Anserini-style defaults):
  first pass   BM25Okapi over whole files (BM25's tuned chunk size), canonical tokenizer
  feedback     fbDocs = 10, fbTerms = 10, originalQueryWeight = 0.5
  RM1          P(t|R) ∝ sum_d P(t|d) * s(d), s = first-pass scores normalized over the top 10
  stop terms   tokens in more than half of the repository's files are not expansion terms
  final score  0.5 * sum_{t in q} w_q(t) bm25_t(d) + 0.5 * sum_{t in E} w_E(t) bm25_t(d)

Runs on the same query sets and corpus snapshots as the genarm control arm.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import run_retrieval as S  # noqa: E402

FB_DOCS, FB_TERMS, ORIG_W = 10, 10, 0.5


def run_repo(qf: Path, clone: Path) -> list[dict]:
    from rank_bm25 import BM25Okapi
    paths = S.list_corpus(clone)
    # exactly the canonical whole-file document: chunk_file(text, c=1e9) yields one chunk
    # (or "# empty file"), so the first pass reproduces bm25_sweep202610 whole-file BM25.
    docs = [S.tokenize(S.chunk_file(S.read_file(clone, p), 1_000_000_000, 10)[0]) for p in paths]
    bm25 = BM25Okapi(docs)
    tf = [Counter(d) for d in docs]
    dlen = np.array([len(d) for d in docs], dtype=float)
    df = Counter(t for c in tf for t in c)
    stop = {t for t, n in df.items() if n > 0.5 * len(docs)}
    term_cache: dict[str, np.ndarray] = {}

    def term_scores(t: str) -> np.ndarray:
        if t not in term_cache:
            term_cache[t] = bm25.get_scores([t])
        return term_cache[t]

    out = []
    for q in S.load_qrels(qf, 100000, 0, 3):
        text = S.strip_pr_boilerplate(q["problem_statement"])
        qt = S.tokenize(text)
        base = bm25.get_scores(qt)
        top = np.argsort(-base)[:FB_DOCS]
        s = base[top] - base[top].min() if len(top) > 1 else np.ones(len(top))
        s = s / s.sum() if s.sum() > 0 else np.full(len(top), 1 / len(top))
        rel: Counter = Counter()
        for w, i in zip(s, top):
            for t, c in tf[i].items():
                if t not in stop:
                    rel[t] += w * c / dlen[i]
        exp = rel.most_common(FB_TERMS)
        z = sum(v for _, v in exp) or 1.0
        qc = Counter(qt)
        qz = sum(qc.values()) or 1
        score = np.zeros(len(docs))
        for t, c in qc.items():
            score += ORIG_W * (c / qz) * term_scores(t)
        for t, v in exp:
            score += (1 - ORIG_W) * (v / z) * term_scores(t)
        order = np.argsort(-score)[:10]
        b10 = [paths[i] for i in np.argsort(-base)[:10]]
        r10 = [paths[i] for i in order]
        gold = set(q["gold_files"])
        out.append({"query_id": q["query_id"], "gold_files": q["gold_files"],
                    "bm25_top10": b10, "rm3_top10": r10,
                    "bm25_ndcg10": S.ndcg_at_k(b10, gold, 10), "rm3_ndcg10": S.ndcg_at_k(r10, gold, 10),
                    "expansion_terms": [t for t, _ in exp]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qrels-root", type=Path, required=True, help="dir with qrels_cell{X}_expanded/")
    ap.add_argument("--repos", type=Path, required=True)
    ap.add_argument("--repos-2024q1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for cell in ("A", "B", "RA", "RB"):
        root = args.repos_2024q1 if cell == "RA" else args.repos
        for qf in sorted((args.qrels_root / f"qrels_cell{cell}_expanded").glob("*.jsonl")):
            dest = args.out / f"{cell}__{qf.stem}.jsonl"
            if dest.is_file() or not (root / qf.stem).is_dir():
                continue
            rows = run_repo(qf, root / qf.stem)
            dest.write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(f"[done] {cell} {qf.stem}: {len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
