#!/usr/bin/env python3
"""Learned-sparse baseline: SPLADE on the genarm cells.

Model: naver/splade-cocondenser-ensembledistil (open SPLADE++ checkpoint).
Documents: the canonical c=1000 chunks (10-line overlap) of each repository,
max-pooled to files exactly as for dense retrieval; SPLADE's 512-token input
limit rules out whole files. Representation: max over tokens of
log(1 + ReLU(MLM logits)), masked; score = sparse dot product.

Queries: the same per-cell query sets as the generative arm, both the
original (boilerplate-stripped) text and the text plus generated anchors, so
SPLADE also gets the "give the anchors to the other retriever" control.

Output per (cell, repo): orig_top10 / expanded_top10 per query.
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import run_retrieval as S  # noqa: E402

MODEL = "naver/splade-cocondenser-ensembledistil"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from scipy import sparse
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForMaskedLM.from_pretrained(MODEL).cuda().eval().half()
    vocab = model.config.vocab_size

    @torch.no_grad()
    def encode(texts: list[str]):
        rows, cols, vals = [], [], []
        for i in range(0, len(texts), args.batch):
            b = tok(texts[i:i + args.batch], padding=True, truncation=True,
                    max_length=args.max_len, return_tensors="pt").to("cuda")
            logits = model(**b).logits
            w = torch.log1p(torch.relu(logits.float())) * b["attention_mask"].unsqueeze(-1)
            v = w.max(dim=1).values  # (batch, vocab)
            nz = v.nonzero(as_tuple=False).cpu().numpy()
            rows.append(nz[:, 0] + i)
            cols.append(nz[:, 1])
            vals.append(v[v > 0].cpu().numpy())
        return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(len(texts), vocab), dtype=np.float32)

    snap = {"A": "repos", "B": "repos", "RB": "repos", "RA": "repos_2024q1"}
    doc_cache: dict[tuple[str, str], tuple] = {}
    for cell, root in snap.items():
        for qf in sorted((args.data / "genarm" / f"qrels_cell{cell}_expanded").glob("*.jsonl")):
            repo = qf.stem
            dest = args.out / f"{cell}__{repo}.jsonl"
            clone = args.data / root / repo
            if dest.is_file() or not clone.is_dir():
                continue
            key = (root, repo)
            if key not in doc_cache:
                paths = S.list_corpus(clone)
                chunks, ranges = [], []
                for p in paths:
                    cs = S.chunk_file(S.read_file(clone, p), 1000, 10)
                    ranges.append((len(chunks), len(chunks) + len(cs)))
                    chunks.extend(cs)
                doc_cache.clear()  # one repository's index in memory at a time
                doc_cache[key] = (paths, ranges, encode(chunks).T.tocsr())
            paths, ranges, D = doc_cache[key]
            qs = S.load_qrels(qf, 100000, 0, 3)
            out = []
            for field, src in (("orig", "problem_statement_stripped"), ("expanded", "expanded_query")):
                Q = encode([q[src] for q in qs])
                scores = (Q @ D).toarray()  # (n_queries, n_chunks)
                for i, q in enumerate(qs):
                    fs = np.array([scores[i, s:e].max() if e > s else 0.0 for s, e in ranges])
                    if field == "orig":
                        out.append({"query_id": q["query_id"], "gold_files": q["gold_files"]})
                    out[i][f"{field}_top10"] = [paths[j] for j in np.argsort(-fs)[:10]]
            dest.write_text("".join(json.dumps(r) + "\n" for r in out))
            print(f"[done] {cell} {repo}: {len(out)} queries", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
