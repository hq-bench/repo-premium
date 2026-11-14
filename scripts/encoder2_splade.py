#!/usr/bin/env python3
"""Second encoder (Qwen3-Embedding-8B), SPLADE and ColBERTv2 on the genarm cells.

Per cell, with BM25 on whole files (BM25's tuned chunk size):
  qwen8: premium_tuned = dense(orig) - BM25(orig); residual = dense(orig) - BM25(query+anchors);
         hyde_gap = BM25 gain from anchors - dense gain from anchors
  splade: splade(orig), splade(query+anchors), and the canonical dense premium over SPLADE
Guard: Qwen3-8B on the original query must reproduce the cached canonical qwen8 lists (A, B).
Repository-cluster bootstrap (10,000, seed 20260813).
"""
from paths import DATA_ROOT
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_model import ndcg10  # noqa: E402
from genarm_rebase import DENSE, WHOLE, control_whole, gen_dir, load  # noqa: E402

SM = (DATA_ROOT / 'smoke_test_results')


def side(d: Path, cell: str) -> dict:
    out = {}
    for f in d.glob(f"{cell}__*.jsonl"):
        repo = f.stem[len(cell) + 2:]
        for line in f.read_text().splitlines():
            r = json.loads(line)
            out[(repo, r["query_id"])] = r
    return out


def boot(by, n=10000, seed=20260813):
    rng = random.Random(seed)
    g = list(by.values())
    d = sorted(sum(s) / len(s) for s in ([x for _ in g for x in g[rng.randrange(len(g))]] for _ in range(n)))
    return [d[int(.025 * n)], d[int(.975 * n)]]


res = {}
for cell in ("A", "B", "RA", "RB"):
    q8, sp = side(SM / "hyde_qwen8_202610", cell), side(SM / "splade_202610", cell)
    cb = side(SM / "colbert_202610", cell)
    jd, bw, gw = load(DENSE[cell]), control_whole(cell), load(gen_dir(cell, WHOLE))
    canon8 = load(SM / f"canon202607/qwen8_cell{cell}_c1000") if cell in ("A", "B") else {}
    keys = sorted(set(q8) & set(sp) & set(cb) & set(jd) & set(bw) & set(gw))
    guard = {"n": len(keys)}
    if canon8:
        ks = [k for k in keys if k in canon8]
        guard["qwen8_identical"] = sum(q8[k]["orig_top10"] == canon8[k]["dense_top10"] for k in ks)
        guard["qwen8_compared"] = len(ks)
        guard["capped"] = sum("max_seq_length_cap" in q8[k] for k in keys)
    by = defaultdict(lambda: defaultdict(list))
    means = defaultdict(float)
    for k in keys:
        gold = jd[k]["gold_files"]
        q0, q1 = ndcg10(q8[k]["orig_top10"], gold), ndcg10(q8[k]["expanded_top10"], gold)
        s0, s1 = ndcg10(sp[k]["orig_top10"], gold), ndcg10(sp[k]["expanded_top10"], gold)
        b0, b1, j0 = bw[k]["bm25_ndcg10"], gw[k]["bm25_ndcg10"], jd[k]["dense_ndcg10"]
        c0, c1 = ndcg10(cb[k]["orig_top10"], gold), ndcg10(cb[k]["expanded_top10"], gold)
        vals = {"qwen8_premium_tuned": q0 - b0, "qwen8_residual_gen": q0 - b1,
                "qwen8_hyde_gap": (b1 - b0) - (q1 - q0), "qwen8_dense_gain": q1 - q0,
                "splade_minus_bm25": s0 - b0, "jina_minus_splade": j0 - s0,
                "splade_gain_from_anchors": s1 - s0, "splade_exp_minus_bm25_gen": s1 - b1,
                "colbert_minus_bm25": c0 - b0, "jina_minus_colbert": j0 - c0,
                "colbert_gain_from_anchors": c1 - c0, "colbert_exp_minus_bm25_gen": c1 - b1}
        for n, v in vals.items():
            by[n][k[0]].append(v)
        for n, v in (("qwen8", q0), ("qwen8_exp", q1), ("splade", s0), ("splade_exp", s1),
                     ("colbert", c0), ("colbert_exp", c1),
                     ("bm25_whole", b0), ("bm25_gen", b1), ("jina", j0)):
            means[n] += v / len(keys)
    res[cell] = {"guard": guard, "means": dict(means),
                 **{n: {"mean": sum(x for g in by[n].values() for x in g) / len(keys), "ci95": boot(by[n])} for n in by}}
    c = res[cell]
    print(f"{cell:2s} guard {guard} | " + " ".join(f"{k}={v:.3f}" for k, v in means.items()))
    for n in by:
        print(f"     {n:34s} {c[n]['mean']:+.3f} [{c[n]['ci95'][0]:+.3f},{c[n]['ci95'][1]:+.3f}]")
(DATA_ROOT / 'analysis/encoder2_splade.json').write_text(json.dumps(res, indent=2) + "\n")
