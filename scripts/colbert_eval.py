#!/usr/bin/env python3
"""Late-interaction baseline: ColBERTv2 on the genarm cells (exhaustive MaxSim).

Checkpoint colbert-ir/colbertv2.0 (BERT-base + linear 768->128), loaded with
plain transformers: query marker [unused0], document marker [unused1] after
[CLS], token embeddings projected to 128 dims and L2-normalised; score =
sum over query tokens of the max similarity over document tokens.

Documents: the canonical c=1000 chunks (10-line overlap), max-pooled to
files exactly as for dense retrieval and SPLADE; doc_maxlen 300 tokens.
Queries: ColBERT's default query_maxlen of 32 would cut PR-length queries to
a sentence, so queries are truncated at 256 tokens; queries shorter than 32
tokens are padded with [MASK] to 32 (ColBERT's query augmentation).
Both the original and the anchor-expanded query are scored.
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

CKPT = "colbert-ir/colbertv2.0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--doc-maxlen", type=int, default=300)
    ap.add_argument("--query-maxlen", type=int, default=256)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, BertModel

    path = snapshot_download(CKPT)
    tok = AutoTokenizer.from_pretrained(path)
    bert = BertModel.from_pretrained(path, add_pooling_layer=False).cuda().eval().half()
    lin = torch.nn.Linear(768, 128, bias=False)
    lin.weight.data = load_file(f"{path}/model.safetensors")["linear.weight"]
    lin = lin.cuda().half()
    qm, dm = tok.convert_tokens_to_ids("[unused0]"), tok.convert_tokens_to_ids("[unused1]")
    mask_id = tok.mask_token_id

    def ids(texts, marker, maxlen, augment):
        enc = tok(texts, add_special_tokens=False, truncation=True, max_length=maxlen - 3)["input_ids"]
        seqs = []
        for e in enc:
            s = [tok.cls_token_id, marker] + e + [tok.sep_token_id]
            if augment and len(s) < 32:
                s += [mask_id] * (32 - len(s))
            seqs.append(s)
        return seqs

    @torch.no_grad()
    def embed(seqs, bs):
        out = []
        for i in range(0, len(seqs), bs):
            b = seqs[i:i + bs]
            L = max(len(s) for s in b)
            x = torch.tensor([s + [0] * (L - len(s)) for s in b], device="cuda")
            m = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in b], device="cuda")
            h = lin(bert(input_ids=x, attention_mask=m).last_hidden_state)
            h = torch.nn.functional.normalize(h.float(), dim=-1).half() * m.unsqueeze(-1)
            out.extend(h[j, :len(b[j])] for j in range(len(b)))
        return out

    snap = {"A": "repos", "B": "repos", "RB": "repos", "RA": "repos_2024q1"}
    cache: dict = {}
    for cell, root in snap.items():
        for qf in sorted((args.data / "genarm" / f"qrels_cell{cell}_expanded").glob("*.jsonl")):
            repo = qf.stem
            dest = args.out / f"{cell}__{repo}.jsonl"
            clone = args.data / root / repo
            if dest.is_file() or not clone.is_dir():
                continue
            key = (root, repo)
            if key not in cache:
                cache.clear()
                paths = S.list_corpus(clone)
                chunks, ranges = [], []
                for p in paths:
                    cs = S.chunk_file(S.read_file(clone, p), 1000, 10)
                    ranges.append((len(chunks), len(chunks) + len(cs)))
                    chunks.extend(cs)
                D = embed(ids(chunks, dm, args.doc_maxlen, False), args.batch)
                Lmax = max(d.shape[0] for d in D)
                Dpad = torch.zeros(len(D), Lmax, 128, dtype=torch.half, device="cuda")
                for i, d in enumerate(D):
                    Dpad[i, :d.shape[0]] = d
                cache[key] = (paths, ranges, Dpad)
            paths, ranges, Dpad = cache[key]
            qs = S.load_qrels(qf, 100000, 0, 3)
            out = [{"query_id": q["query_id"], "gold_files": q["gold_files"]} for q in qs]
            for field, src in (("orig", "problem_statement_stripped"), ("expanded", "expanded_query")):
                Q = embed(ids([q[src] for q in qs], qm, args.query_maxlen, True), 32)
                for i, qv in enumerate(Q):
                    sc = torch.empty(Dpad.shape[0], device="cuda")
                    for s in range(0, Dpad.shape[0], 4096):  # chunked MaxSim to bound memory
                        sim = torch.einsum("qd,nld->nql", qv.float(), Dpad[s:s + 4096].float())
                        sc[s:s + 4096] = sim.max(dim=2).values.sum(dim=1)
                    c = sc.cpu().numpy()
                    fs = np.array([c[a:b].max() if b > a else -1e9 for a, b in ranges])
                    out[i][f"{field}_top10"] = [paths[j] for j in np.argsort(-fs)[:10]]
            dest.write_text("".join(json.dumps(r) + "\n" for r in out))
            print(f"[done] {cell} {repo}: {len(out)} queries", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
