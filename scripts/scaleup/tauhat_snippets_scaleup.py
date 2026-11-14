#!/usr/bin/env python3
"""Query-only tau-hat and top-5 file snippets for the scale-up (decision model).

For each scale-up task, rebuild the corpus exactly as the retrieval did
(ls-tree at the base commit, CODE_EXTS filter, c=1000 / 10-line chunks,
camelCase tokenization for go/java/typescript_javascript), fit the
paper-canonical Router (embedding_premium.Router.fit -> tau_query: top-5
mean IDF of the stripped query's tokens; no gold information), and record
tau_hat per query. It also emits, for every file in the union of the dense
and BM25 top-5 lists of the jina-0.5b run, a short snippet (first 8
non-blank lines, <= 400 chars) for the zero-shot LLM decision model.

Nothing is written except --out. Blobs stream from the bare mirror.

Run on the GPU host (copy local embedding_premium.py + smoke_test_django.py into
--smoke first; the server checkout lacks embedding_premium.py):
  python tauhat_snippets_scaleup.py \
     --manifest .../experiment_v1/audit_v2/eligible_main.tsv \
     --retrieval .../experiment_v1/retrieval/jina-0.5b \
     --smoke ${REPO_PREMIUM_DATA:-data}/decision_model --out tauhat_snippets_scaleup.jsonl
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
S = None
EP = None
RETR = None


def _init(smoke_dir: str, retr: str) -> None:
    global S, EP, RETR
    sys.path.insert(0, smoke_dir)
    import run_retrieval as s  # noqa: E402
    import embedding_premium as ep  # noqa: E402
    S, EP, RETR = s, ep, Path(retr)


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
            t = payload.decode("utf-8", errors="ignore")
            texts[path] = t.replace("\r\n", "\n").replace("\r", "\n")
    finally:
        p.stdin.close(); p.wait()
    return texts


def snippet(text: str) -> str:
    lines = [l.rstrip() for l in text.split("\n") if l.strip()][:8]
    return "\n".join(lines)[:400]


def one_task(row: dict) -> list[dict]:
    S.TOKENIZE_CAMELCASE = row["language"] in CAMEL_LANGS
    exts = tuple(e.strip() if e.strip().startswith(".") else "." + e.strip()
                 for e in row["code_exts"].split(",") if e.strip())
    texts = corpus_texts(row["corpus_ref"], row["base_commit"], exts)
    chunks = []
    for pth in sorted(texts):
        chunks.extend(S.chunk_file(texts[pth], 1000, 10))
    router = EP.Router.fit(chunks)
    lists = {}
    rf = RETR / row["task_id"] / "smoke_test_django.jsonl"
    if rf.exists():
        for l in rf.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                lists[r["query_id"]] = r
    out = []
    qrels = [json.loads(l) for l in Path(row["qrel_file"]).read_text().splitlines() if l.strip()]
    for q in qrels:
        r = lists.get(q["query_id"], {})
        files = list(dict.fromkeys(r.get("dense_top10", [])[:5] + r.get("bm25_top10", [])[:5]))
        out.append({"query_id": q["query_id"], "task_id": row["task_id"],
                    "language": row["language"],
                    "tau_hat": router.tau_query(q["problem_statement"]),
                    "snippets": {f: snippet(texts.get(f, "")) for f in files}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--smoke", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    with open(a.manifest) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    outp = Path(a.out)
    done = set()
    if outp.exists():
        done = {json.loads(l)["task_id"] for l in outp.read_text().splitlines() if l.strip()}
    todo = [r for r in rows if r["task_id"] not in done]
    print(f"{len(rows)} tasks, {len(todo)} to do", flush=True)
    with Pool(a.workers, initializer=_init, initargs=(a.smoke, a.retrieval)) as pool, \
            outp.open("a") as fo:
        for i, recs in enumerate(pool.imap_unordered(one_task, todo), 1):
            for r in recs:
                fo.write(json.dumps(r) + "\n")
            fo.flush()
            if i % 20 == 0:
                print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
