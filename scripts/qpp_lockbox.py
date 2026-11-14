#!/usr/bin/env python3
"""Retrieval scores + query features for the LOCKBOX (decision model v2).

Runs only after the serve model is frozen (analysis/decision_model_v2_artifacts).
For every lockbox query it records exactly what the frozen model needs and
nothing about relevance beyond the gold list the evaluation reads:

  bm25     BM25 over whole files (pool's extensions and tokenizer), top-50 + corpus mean/std
  dense    jina-code-0.5b, c=1000 chunks max-pooled per file (pool's max_seq_length),
           top-50 + corpus mean/std
  tau_hat  top-5 mean IDF of the stripped query over c=1000 chunks — the
           embedding_premium.Router.tau_query definition used on dev
  query    raw query text (features strip it themselves, as on dev)

Each manifest row mirrors the pool's canonical runner flags (R9 *_rerun.slurm,
DeepSWE retrieval.slurm). Manifest TSV columns:
  pool  key  qrel_file  clone_dir  code_exts  camel  max_seq_length  strip  n  max_gold  batch
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import run_retrieval as S  # noqa: E402

TOPN = 50
WHOLE = 1_000_000_000


def top(scores: np.ndarray, paths: list[str]) -> list[list]:
    order = np.argsort(-scores)[:TOPN]
    return [[paths[i], float(scores[i])] for i in order]


def run_row(row: dict, model, out_dir: Path, batch: int) -> None:
    dest = out_dir / f"{row['pool']}__{row['key']}.jsonl"
    if dest.is_file():
        return
    clone = Path(row["clone_dir"])
    exts = tuple(e.strip() for e in row["code_exts"].split(",") if e.strip())
    qrels = S.load_qrels(Path(row["qrel_file"]), int(row["n"]), 0, int(row["max_gold"]))
    paths = S.list_corpus(clone, exts)
    texts = [S.read_file(clone, p) for p in paths]

    c_chunks, c_ranges = [], []
    for t in texts:
        cs = S.chunk_file(t, 1000, 10)
        c_ranges.append((len(c_chunks), len(c_chunks) + len(cs)))
        c_chunks.extend(cs)

    # tau_hat: Router.fit/tau_query on c=1000 chunks, default (non-camel) tokenizer
    S.TOKENIZE_CAMELCASE = False
    idf = S.compute_idf([S.tokenize(c) for c in c_chunks])

    def tau_hat(q: str) -> float:
        toks = S.tokenize(S.strip_pr_boilerplate(q))
        vals = sorted((idf.get(t, 0.0) for t in toks), reverse=True)[:5]
        return float(sum(vals) / len(vals)) if vals else 0.0

    # retrieval tokenizer = the pool's
    S.TOKENIZE_CAMELCASE = row["camel"] == "1"
    w_chunks, w_ranges = [], []
    for t in texts:
        cs = S.chunk_file(t, WHOLE, 10)
        w_ranges.append((len(w_chunks), len(w_chunks) + len(cs)))
        w_chunks.extend(cs)
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([S.tokenize(c) for c in w_chunks])

    if int(row["max_seq_length"]) > 0:
        model.max_seq_length = int(row["max_seq_length"])
    d_emb = model.encode(c_chunks, batch_size=int(row.get("batch") or batch), normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)

    out = []
    for q in qrels:
        raw = q["problem_statement"]
        text = S.strip_pr_boilerplate(raw) if row["strip"] == "1" else raw
        cs = bm25.get_scores(S.tokenize(text))
        b = np.array([cs[s:e].max() if e > s else 0.0 for s, e in w_ranges])
        qe = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = d_emb @ qe
        d = np.array([sims[s:e].max() if e > s else -1.0 for s, e in c_ranges])
        out.append({"pool": row["pool"], "key": row["key"], "query_id": q["query_id"],
                    "gold_files": q["gold_files"], "query": raw, "tau_hat": tau_hat(raw),
                    "bm25": top(b, paths), "bm25_mean": float(b.mean()), "bm25_std": float(b.std()),
                    "dense": top(d, paths), "dense_mean": float(d.mean()), "dense_std": float(d.std()),
                    "n_files": len(paths)})
    S.TOKENIZE_CAMELCASE = False
    dest.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"[done] {row['pool']} {row['key']}: {len(out)} queries, {len(c_chunks)} chunks", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("jinaai/jina-code-embeddings-0.5b", trust_remote_code=True,
                                device=args.device)
    rows = list(csv.DictReader(args.manifest.open(), delimiter="\t"))
    if args.limit:
        rows = rows[: args.limit]
    for row in rows:
        try:
            run_row(row, model, args.out, args.batch)
        except Exception as e:  # one bad snapshot must not sink the array
            print(f"[fail] {row['pool']} {row['key']}: {e!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
