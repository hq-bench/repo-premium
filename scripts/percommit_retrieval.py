#!/usr/bin/env python3
"""Retrieval with each task's own pre-change commit as the corpus.

The paper's core pools share one corpus snapshot per repository, which already contains
most curated fixes and predates the 2026 raw changes by years. This runner re-scores every
query against the repository at its own base commit (the parent of the resolving change).

For one repository it walks the distinct commits in the task file, lists the indexed files
(`git ls-tree`, `.py`), reads each blob once (`git cat-file --batch`) and keys all derived
state by blob SHA, so a file version shared by many commits is chunked, tokenized and
embedded once. Per commit it fits chunk-level BM25 (c=1000, 10-line overlap) and whole-file
BM25, max-pools dense chunk similarities per file (canonical jina-code-embeddings-0.5b, no
instruction prefix), and scores that commit's queries. Retrieval mirrors
scripts/smoke_test_django.py exactly (same chunker, tokenizer, BM25Okapi,
normalized embeddings, file max-pool).

Guard: tasks with pool `guard-*` are the Cur-Pre/Raw-Post queries evaluated at the original
snapshot commit, where results must match the canonical runs.

Usage (inside the container):
  python percommit_retrieval.py --repo-dir <clone with history> --tasks <repo>.jsonl --out <dir>
Writes <out>/results.jsonl (one record per task) and <out>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_retrieval import chunk_file, tokenize  # noqa: E402

EXTS = (".py",)


class Vocab:
    """Global token -> id map; each document is stored as (ids, counts) in first-occurrence order."""

    def __init__(self):
        self.ids: dict[str, int] = {}

    def doc(self, toks: list[str]) -> tuple[np.ndarray, np.ndarray]:
        freqs: dict[str, int] = {}
        for w in toks:
            freqs[w] = freqs.get(w, 0) + 1
        ids = [self.ids.setdefault(w, len(self.ids)) for w in freqs]
        return np.array(ids, np.int64), np.array(list(freqs.values()), np.int64)


class FastBM25:
    """Vectorized rank_bm25.BM25Okapi (k1=1.5, b=0.75, epsilon=0.25) with identical scores.

    rank_bm25 scores each query token with a Python loop over every document; on sympy's
    multi-megabyte generated files that is minutes per query. This builds a sparse term matrix
    instead, sums idf in rank_bm25's word order (first occurrence scanning documents in order)
    and evaluates the same float expression per query token, so scores match bit for bit.
    """

    def __init__(self, docs: list[tuple[np.ndarray, np.ndarray]], n_vocab: int,
                 k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        from scipy.sparse import csc_matrix
        self.k1, self.b = k1, b
        self.N = len(docs)
        lens = np.array([int(c.sum()) for _, c in docs], np.int64)
        self.doc_len = lens
        self.avgdl = int(lens.sum()) / self.N
        ids = np.concatenate([i for i, _ in docs]); cnt = np.concatenate([c for _, c in docs])
        rows = np.repeat(np.arange(self.N), [len(i) for i, _ in docs])
        self.X = csc_matrix((cnt, (rows, ids)), shape=(self.N, n_vocab))
        nd = np.bincount(ids, minlength=n_vocab)
        uniq, first = np.unique(ids, return_index=True)
        order = uniq[np.argsort(first, kind="stable")]
        idf = {}
        idf_sum = 0
        neg = []
        for w in order.tolist():
            freq = int(nd[w])
            v = math.log(self.N - freq + 0.5) - math.log(freq + 0.5)
            idf[w] = v; idf_sum += v
            if v < 0:
                neg.append(w)
        eps = epsilon * (idf_sum / len(idf))
        for w in neg:
            idf[w] = eps
        self.idf = idf

    def get_scores(self, qids: list[int]) -> np.ndarray:
        score = np.zeros(self.N)
        X = self.X
        for w in qids:
            v = self.idf.get(w) or 0
            if not v:
                continue
            q_freq = np.zeros(self.N, np.int64)
            s, e = X.indptr[w], X.indptr[w + 1]
            q_freq[X.indices[s:e]] = X.data[s:e]
            score += v * (q_freq * (self.k1 + 1) / (q_freq + self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)))
        return score


class Blobs:
    """Long-lived `git cat-file --batch` reader."""

    def __init__(self, repo: Path):
        self.p = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def read(self, sha: str) -> bytes:
        self.p.stdin.write((sha + "\n").encode()); self.p.stdin.flush()
        header = self.p.stdout.readline().split()
        if len(header) < 3 or header[1] != b"blob":
            return b""
        size = int(header[2])
        data = self.p.stdout.read(size); self.p.stdout.read(1)
        return data


def ls_tree(repo: Path, commit: str) -> list[tuple[str, str]]:
    out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--full-tree", commit],
                         capture_output=True, text=True, check=True).stdout
    files = []
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        mode, typ, sha = meta.split()
        if typ == "blob" and mode != "120000" and path.endswith(EXTS):
            files.append((path, sha))
    files.sort()
    return files


def ndcg(top: list[str], gold: set[str]) -> float:
    if not gold:
        return float("nan")
    dcg = sum(1 / math.log2(i + 2) for i, f in enumerate(top[:10]) if f in gold)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(len(gold), 10)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="jinaai/jina-code-embeddings-0.5b")
    ap.add_argument("--chunk-chars", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--encode-batch", type=int, default=16)
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    by_commit: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        by_commit[t["commit"]].append(t)
    args.out.mkdir(parents=True, exist_ok=True)
    done = set()
    res_p = args.out / "results.jsonl"
    if res_p.exists():
        done = {(json.loads(l)["pool"], json.loads(l)["query_id"]) for l in res_p.read_text().splitlines() if l.strip()}

    model = SentenceTransformer(args.model, trust_remote_code=True)
    blobs = Blobs(args.repo_dir)
    cache: dict[str, dict] = {}   # blob sha -> {chunks, docs, whole, emb}
    vocab = Vocab()
    stats = {"commits": 0, "blobs_encoded": 0, "chunks_encoded": 0, "missing_commits": []}
    t0 = time.time()
    with res_p.open("a") as fh:
        for ci, (commit, ts) in enumerate(sorted(by_commit.items())):
            ts = [t for t in ts if (t["pool"], t["query_id"]) not in done]
            if not ts:
                continue
            try:
                files = ls_tree(args.repo_dir, commit)
            except subprocess.CalledProcessError:
                stats["missing_commits"].append(commit)
                continue
            # One entry per distinct blob: identical files (e.g. empty __init__.py) share a SHA,
            # and a duplicate here would shift every later embedding slice by one file.
            new = list({s: p for p, s in files if s not in cache}.items())
            new = [(p, s) for s, p in new]
            texts, owners = [], []
            for p, s in new:
                text = blobs.read(s).decode("utf-8", errors="ignore")
                chunks = chunk_file(text, args.chunk_chars, args.overlap)
                cache[s] = {"chunks": chunks, "docs": [vocab.doc(tokenize(c)) for c in chunks],
                            "whole": vocab.doc(tokenize(chunk_file(text, 10**9, args.overlap)[0]))}
                texts.extend(chunks); owners.extend([s] * len(chunks))
            if texts:
                emb = model.encode(texts, batch_size=args.encode_batch, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
                pos = 0
                for s in dict.fromkeys(owners):
                    k = len(cache[s]["chunks"]); cache[s]["emb"] = emb[pos:pos + k]; pos += k
                stats["blobs_encoded"] += len(new); stats["chunks_encoded"] += len(texts)
            paths = [p for p, _ in files]
            ranges, chunk_toks, embs, whole = [], [], [], []
            for _, s in files:
                c = cache[s]; st = len(chunk_toks)
                chunk_toks.extend(c["docs"]); embs.append(c["emb"]); ranges.append((st, st + len(c["docs"])))
                whole.append(c["whole"])
            E = np.concatenate(embs) if embs else np.zeros((0, 1), np.float32)
            bm_chunk = FastBM25(chunk_toks, len(vocab.ids))
            bm_whole = FastBM25(whole, len(vocab.ids))
            path_set = set(paths)
            for t in ts:
                qt = [vocab.ids[w] for w in tokenize(t["query"]) if w in vocab.ids]
                sc = bm_chunk.get_scores(qt)
                f_bm = np.array([sc[a:b].max() if b > a else 0.0 for a, b in ranges])
                f_bw = bm_whole.get_scores(qt)
                q = model.encode([t["query"]], normalize_embeddings=True, convert_to_numpy=True)[0]
                sims = E @ q
                f_de = np.array([sims[a:b].max() if b > a else -1.0 for a, b in ranges])
                top = {k: [paths[i] for i in np.argsort(-v)[:10]] for k, v in (("bm25", f_bm), ("bm25w", f_bw), ("dense", f_de))}
                gold = {g for g in t["gold"] if g in path_set}
                rec = {"pool": t["pool"], "query_id": t["query_id"], "repo": t["repo"], "commit": commit,
                       "gold_files": sorted(gold), "gold_missing": sorted(set(t["gold"]) - gold),
                       "n_corpus_files": len(paths)}
                for k, v in top.items():
                    rec[f"{k}_top10"] = v; rec[f"{k}_ndcg10"] = ndcg(v, gold)
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            stats["commits"] += 1
            if ci % 10 == 0:
                print(f"[{ci}/{len(by_commit)}] {commit[:10]} files={len(files)} new_blobs={len(new)} "
                      f"cache={len(cache)} t={time.time() - t0:.0f}s", flush=True)
    stats["seconds"] = time.time() - t0
    (args.out / "summary.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
