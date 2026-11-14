#!/usr/bin/env python3
"""Measure encoder index-build throughput, for the cost side of sec:genarm.

Generation moves retrieval cost from index build to query time, so the honest
comparison needs both halves measured on the same hardware: chunks per second
for the dense index, against seconds per query for the generator. Reports the
crossover -- the number of queries per repository at which one-off encoding is
cheaper than per-query generation.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from run_retrieval import chunk_file, list_corpus, read_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clone-dir", type=Path, required=True)
    ap.add_argument("--model", default="jinaai/jina-code-embeddings-0.5b")
    ap.add_argument("--limit-chunks", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk-chars", type=int, default=1000)
    ap.add_argument("--gen-seconds-per-query", type=float, default=1.45,
                    help="Measured generator latency (H200, Qwen3-8B, temp 0).")
    args = ap.parse_args()

    paths = list_corpus(args.clone_dir, (".py",))
    chunks: list[str] = []
    for p in paths:
        chunks.extend(chunk_file(read_file(args.clone_dir, p), args.chunk_chars, 10))
        if len(chunks) >= args.limit_chunks:
            break
    chunks = chunks[:args.limit_chunks]
    total_repo_chunks = 0
    for p in paths:
        total_repo_chunks += len(chunk_file(read_file(args.clone_dir, p), args.chunk_chars, 10))

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, trust_remote_code=True, device="cuda")
    model.encode(chunks[:args.batch_size], batch_size=args.batch_size)  # warm up

    t0 = time.time()
    model.encode(chunks, batch_size=args.batch_size, show_progress_bar=False)
    dt = time.time() - t0

    per_chunk = dt / len(chunks)
    repo_build = per_chunk * total_repo_chunks
    crossover = repo_build / args.gen_seconds_per_query
    print(f"model            {args.model}")
    print(f"repo             {args.clone_dir.name}")
    print(f"timed chunks     {len(chunks)} in {dt:.1f}s -> {len(chunks)/dt:.1f} chunks/s "
          f"({per_chunk*1000:.2f} ms/chunk)")
    print(f"repo chunks      {total_repo_chunks}")
    print(f"full index build {repo_build:.0f}s ({repo_build/60:.1f} min)")
    print(f"generator        {args.gen_seconds_per_query:.2f} s/query")
    print(f"CROSSOVER        {crossover:.0f} queries per repository")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
