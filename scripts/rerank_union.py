#!/usr/bin/env python3
"""Off-the-shelf reranker baseline for the decision-model test.

A reranker over the union of the dense and BM25 candidates is an implicit
per-query choice between the two retrievers, so any router has to beat it.
Candidates per query: dense top-10 (jina-code-0.5b, c=1000) union BM25 top-10
(whole files), read from qpp_scores.py output. Documents are the file text
truncated as the canonical runner truncates it (MAX_FILE_CHARS); the model is
jinaai/jina-reranker-v3 with its default query/document token limits.

Output: <out>/<repo>.jsonl with the reranked top-10 and scores per query.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import run_retrieval as S  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True, help="qpp_scores.py output dir")
    ap.add_argument("--clones", type=Path, required=True)
    ap.add_argument("--qrels-a", type=Path, required=True)
    ap.add_argument("--qrels-b", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="jinaai/jina-reranker-v3")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(args.model, dtype="auto", trust_remote_code=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    text = {}
    for cell, root in (("A", args.qrels_a), ("B", args.qrels_b)):
        for f in root.glob("*.jsonl"):
            for q in S.load_qrels(f, 300, 0, 3):
                text[(cell, q["query_id"])] = S.strip_pr_boilerplate(q["problem_statement"])

    for f in sorted(args.scores.glob("*.jsonl")):
        repo = f.stem
        dest = args.out / f"{repo}.jsonl"
        if dest.is_file():
            continue
        clone = args.clones / repo
        rows = []
        for line in f.read_text().splitlines():
            r = json.loads(line)
            cand = list(dict.fromkeys([p for p, _ in r["dense"][:10]] + [p for p, _ in r["bm25"][:10]]))
            docs = [S.read_file(clone, p)[:S.MAX_FILE_CHARS] or "# empty file" for p in cand]
            res = model.rerank(text[(r["cell"], r["query_id"])], docs)
            ranked = [[cand[int(x["index"])], float(x["relevance_score"])] for x in res]
            rows.append({"cell": r["cell"], "query_id": r["query_id"],
                         "gold_files": r["gold_files"], "n_candidates": len(cand),
                         "rerank": ranked[:10]})
        dest.write_text("".join(json.dumps(x) + "\n" for x in rows))
        print(f"[done] {repo}: {len(rows)} queries", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
