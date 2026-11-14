#!/usr/bin/env python3
"""HyDE-style control for the generative arm: does generation help DENSE too?

The generative arm compares BM25 on (query + generated anchors) with dense on
the ORIGINAL query. If the anchors help dense retrieval as much as BM25, the
"premium closes" result says generation helps any retriever rather than that
the premium is missing lexical evidence. This script encodes, per query, the
same texts the BM25 arms used:

  orig       problem_statement_stripped   (guard: must reproduce the cached dense run)
  expanded   expanded_query               (original + anchors; what BM25 got)
  generated  generated_query              (anchors only)

and retrieves with jina-code-0.5b over c=1000 chunks max-pooled per file,
reusing cached chunk embeddings where the corpus snapshot is the same
(qpp_scores.py output for repos/), encoding otherwise (RA: repos_2024q1).
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

FIELDS = ("orig", "expanded", "generated")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_ROOT)
    ap.add_argument("--emb-cache", type=Path, default=None,
                    help="cached chunk embeddings for the repos/ snapshot (jina-code-0.5b only); "
                         "default: encode and cache under --out")
    ap.add_argument("--model", default="jinaai/jina-code-embeddings-0.5b")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer(args.model, trust_remote_code=True, device="cuda")
    default_msl = model.max_seq_length

    snap = {"A": "repos", "B": "repos", "RB": "repos", "RA": "repos_2024q1"}
    for cell, root in snap.items():
        for qf in sorted((args.data / "genarm" / f"qrels_cell{cell}_expanded").glob("*.jsonl")):
            repo = qf.stem
            dest = args.out / f"{cell}__{repo}.jsonl"
            clone = args.data / root / repo
            if dest.is_file() or not clone.is_dir():
                continue
            paths = S.list_corpus(clone)
            chunks, ranges = [], []
            for p in paths:
                cs = S.chunk_file(S.read_file(clone, p), 1000, 10)
                ranges.append((len(chunks), len(chunks) + len(cs)))
                chunks.extend(cs)
            if root == "repos" and args.emb_cache is not None:
                cache = args.emb_cache / f"{repo}.emb.npy"
            else:
                cache = args.out / f"{root}__{repo}.emb.npy"
            if cache.is_file():
                emb = np.load(cache)
                assert emb.shape[0] == len(chunks), (repo, emb.shape, len(chunks))
            else:
                try:
                    emb = model.encode(chunks, batch_size=args.batch, normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=False)
                except torch.OutOfMemoryError:
                    # the canonical sweeps re-ran OOM repos (sympy) with a 2048-token cap
                    torch.cuda.empty_cache()
                    model.max_seq_length = 2048
                    print(f"[oom] {repo}: retrying with max_seq_length=2048", flush=True)
                    emb = model.encode(chunks, batch_size=max(1, args.batch // 2), normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=False)
                np.save(cache, emb.astype(np.float32))
            capped = model.max_seq_length != default_msl
            rows = []
            for q in S.load_qrels(qf, 100000, 0, 3):
                texts = {"orig": q["problem_statement_stripped"], "expanded": q["expanded_query"],
                         "generated": q.get("generated_query") or q["problem_statement_stripped"]}
                rec = {"query_id": q["query_id"], "gold_files": q["gold_files"]}
                for f in FIELDS:
                    qe = model.encode([texts[f]], normalize_embeddings=True, convert_to_numpy=True)[0]
                    sims = emb @ qe
                    fs = np.array([sims[s:e].max() if e > s else -1.0 for s, e in ranges])
                    rec[f"{f}_top10"] = [paths[i] for i in np.argsort(-fs)[:10]]
                rows.append(rec)
            model.max_seq_length = default_msl  # the cap applies to the OOM repo only
            if capped:
                rows = [dict(r, max_seq_length_cap=2048) for r in rows]
            dest.write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(f"[done] {cell} {repo}: {len(rows)} queries", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
