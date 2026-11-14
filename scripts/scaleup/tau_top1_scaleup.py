#!/usr/bin/env python3
"""Recompute the law's frozen predictor (tau_top1 = MaxIDF) on the scale-up.

The scale-up retrieval runs stored only tau_q (= tau_mean5, mean IDF of the
five rarest shared query/gold tokens). The decay law is stated on tau_top1
(IDF of the single rarest shared token), so it is recomputed here with the
SAME helpers the retrieval used (the server copy of smoke_test_django.py,
imported, not reimplemented) over the SAME corpus (ls-tree at the task's base
commit, filtered by CODE_EXTS exactly as list_corpus does), with chunk-level
IDF at --chunk-chars 1000 / 10-line overlap and camelCase tokenization for
go/java/typescript_javascript, matching retrieval.slurm.

Nothing is written to the shared filesystem except --out. Blobs are streamed
from the bare mirror into memory; no worktree is materialized.

Guard: the recomputed tau_mean5 must equal the stored tau_q for every query;
the analysis script refuses to run otherwise.

Run on the GPU host:
  python tau_top1_scaleup.py \
     --manifest .../audit_v2/eligible_main.tsv \
     --smoke scripts \
     --out ${REPO_PREMIUM_DATA:-data}/law_confirm/tau_top1_scaleup.jsonl --workers 16
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

BLOB_MODES = {b"100644", b"100755"}
CAMEL_LANGS = {"go", "java", "typescript_javascript"}
SMOKE = None


def _init(smoke_dir: str) -> None:
    global SMOKE
    sys.path.insert(0, smoke_dir)
    import run_retrieval as s  # noqa: E402
    SMOKE = s


def read_exactly(stream, n: int) -> bytes:
    out, rem = [], n
    while rem > 0:
        b = stream.read(rem)
        if not b:
            raise RuntimeError("EOF")
        out.append(b)
        rem -= len(b)
    return b"".join(out)


def corpus_texts(mirror: str, commit: str, exts: tuple[str, ...]) -> dict[str, str]:
    ls = subprocess.run(["git", "-C", mirror, "ls-tree", "-r", "-z", commit],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    entries = []
    for rec in ls.stdout.split(b"\0"):
        meta, _, raw = rec.partition(b"\t")
        if not raw:
            continue
        parts = meta.split(b" ")
        if len(parts) < 3 or parts[0] not in BLOB_MODES:
            continue
        path = raw.decode("utf-8", errors="surrogateescape")
        if path.endswith(exts):
            entries.append((parts[2].decode(), path))
    texts: dict[str, str] = {}
    p = subprocess.Popen(["git", "-C", mirror, "cat-file", "--batch"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for oid, path in entries:
            p.stdin.write((oid + "\n").encode()); p.stdin.flush()
            hdr = p.stdout.readline().split()
            payload = read_exactly(p.stdout, int(hdr[2]))
            p.stdout.read(1)
            # Path.read_text(errors="ignore") = utf-8 decode + universal newlines
            t = payload.decode("utf-8", errors="ignore")
            texts[path] = t.replace("\r\n", "\n").replace("\r", "\n")
    finally:
        p.stdin.close(); p.wait()
    return texts


def one_task(row: dict) -> list[dict]:
    s = SMOKE
    s.TOKENIZE_CAMELCASE = row["language"] in CAMEL_LANGS
    exts = tuple(e.strip() if e.strip().startswith(".") else "." + e.strip()
                 for e in row["code_exts"].split(",") if e.strip())
    texts = corpus_texts(row["corpus_ref"], row["base_commit"], exts)
    paths = sorted(texts)
    chunk_tokens, ranges = [], {}
    for pth in paths:
        cs = s.chunk_file(texts[pth], 1000, 10)
        st = len(chunk_tokens)
        chunk_tokens.extend(s.tokenize(c) for c in cs)
        ranges[pth] = (st, len(chunk_tokens))
    idf = s.compute_idf(chunk_tokens)
    out = []
    qrels = [json.loads(l) for l in Path(row["qrel_file"]).read_text().splitlines() if l.strip()]
    for q in qrels:
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
            shared = set()
        out.append({"query_id": q["query_id"], "task_id": row["task_id"],
                    "language": row["language"], "n_files": len(paths),
                    "n_chunks": len(chunk_tokens), "n_shared": len(shared),
                    "tau_mean5": mean5, "tau_top1": top1})
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
    done = set()
    outp = Path(a.out)
    if outp.exists():
        done = {json.loads(l)["task_id"] for l in outp.read_text().splitlines() if l.strip()}
    todo = [r for r in rows if r["task_id"] not in done]
    print(f"{len(rows)} tasks, {len(todo)} to do", flush=True)
    with Pool(a.workers, initializer=_init, initargs=(a.smoke,)) as pool, outp.open("a") as fo:
        for i, recs in enumerate(pool.imap_unordered(one_task, todo), 1):
            for r in recs:
                fo.write(json.dumps(r) + "\n")
            fo.flush()
            if i % 20 == 0:
                print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
