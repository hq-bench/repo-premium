#!/usr/bin/env python3
"""Curated vs raw premium outside Python, against BM25 at its own best chunk size.

Slices (BM25 sweep: smoke_test_results/ml_sweep202610/<slice>_c<C>/<repo>/;
dense: the cached canonical jina-code-0.5b c=1000 run of the same slice):

  TypeScript  cur_vuejs_ts, raw_vuejs_ts (same repo and snapshot), cur_mui_ts, cur_darkreader_ts
  Rust        cur_tokio_rs, raw_tokio_rs (same repo and snapshot)

BM25's best chunk size is chosen per language, pooled over that language's
curated and raw slices, so both sides of a contrast share one setting
(choosing it on these queries favours BM25). Guard: BM25 at c=1000 must
reproduce each cached run's BM25 exactly (tie-order reorders of an identical
top-10 set are counted, not failed). CIs: query-level paired bootstrap
(single repositories), 10,000 resamples, seed 0.
"""
from __future__ import annotations
from paths import DATA_ROOT

import json
import random
from pathlib import Path

SMOKE = (DATA_ROOT / 'smoke_test_results')
SWEEP = SMOKE / "ml_sweep202610"
GRID = (1000, 3000, 1000000000)
SLICES = {  # name: (language, repo, curated?, cached canonical dense dir)
    "cur_vuejs_ts": ("ts", "vuejs__core", True, "cur_vuejs_ts_jina05_c1000"),
    "raw_vuejs_ts": ("ts", "vuejs__core", False, "r9_vuejs_2023_jina-0.5b_c1000"),
    "cur_mui_ts": ("ts", "mui__material-ui", True, "cur_mui_ts_jina05_c1000"),
    "cur_darkreader_ts": ("ts", "darkreader__darkreader", True, "cur_darkreader_ts_jina05_c1000"),
    "cur_tokio_rs": ("rs", "tokio-rs__tokio", True, "cur_tokio_rs_jina05_c1000"),
    "raw_tokio_rs": ("rs", "tokio-rs__tokio", False, "r9d_rust_c1000"),
}


def load(f: Path) -> dict[str, dict]:
    return {json.loads(l)["query_id"]: json.loads(l) for l in f.read_text().splitlines() if l.strip()}


def ci(v: list[float], n: int = 10000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    m = sorted(sum(v[rng.randrange(len(v))] for _ in v) / len(v) for _ in range(n))
    return [m[int(.025 * n)], m[int(.975 * n)]]


def main() -> int:
    data, guards = {}, {}
    for name, (lang, repo, cur, ddir) in SLICES.items():
        dense = load(SMOKE / ddir / repo / "smoke_test_django.jsonl")
        bm = {c: load(SWEEP / f"{name}_c{c}" / repo / "smoke_test_django.jsonl") for c in GRID}
        off = [k for k in dense if abs(bm[1000][k]["bm25_ndcg10"] - dense[k]["bm25_ndcg10"]) > 1e-12]
        real = [k for k in off if set(bm[1000][k]["bm25_top10"]) != set(dense[k]["bm25_top10"])]
        guards[name] = {"n": len(dense), "n_sweep": len(bm[1000]), "tie_reorders": len(off) - len(real),
                        "real_diffs": len(real)}
        if real or len(bm[1000]) != len(dense):
            raise SystemExit(f"guard failed {name}: {guards[name]}")
        data[name] = (dense, bm)
    out = {"guards": guards, "best_c": {}, "slices": {}, "contrasts": {}}
    for lang in ("ts", "rs"):
        names = [n for n in SLICES if SLICES[n][0] == lang]
        pooled = {c: [r["bm25_ndcg10"] for n in names for r in data[n][1][c].values()] for c in GRID}
        out["best_c"][lang] = max(GRID, key=lambda c: sum(pooled[c]) / len(pooled[c]))
        out["best_c"][lang + "_curve"] = {c: sum(v) / len(v) for c, v in pooled.items()}
    for name, (dense, bm) in data.items():
        c = out["best_c"][SLICES[name][0]]
        d = [dense[k]["dense_ndcg10"] - bm[c][k]["bm25_ndcg10"] for k in dense]
        d0 = [dense[k]["dense_ndcg10"] - bm[1000][k]["bm25_ndcg10"] for k in dense]
        out["slices"][name] = {"n": len(d), "best_c": c, "premium_c1000": sum(d0) / len(d0),
                               "premium_tuned": sum(d) / len(d), "ci95": ci(d), "_v": d}
    for lang, cur, raw in (("ts", "cur_vuejs_ts", "raw_vuejs_ts"), ("rs", "cur_tokio_rs", "raw_tokio_rs")):
        a, b = out["slices"][cur]["_v"], out["slices"][raw]["_v"]
        rng = random.Random(0)
        diffs = sorted(sum(b[rng.randrange(len(b))] for _ in b) / len(b)
                       - sum(a[rng.randrange(len(a))] for _ in a) / len(a) for _ in range(10000))
        out["contrasts"][f"{raw} - {cur}"] = {"diff": sum(b) / len(b) - sum(a) / len(a),
                                             "ci95": [diffs[250], diffs[9750]]}
    for s in out["slices"].values():
        s.pop("_v")
    (DATA_ROOT / 'analysis/ml_rebase.json').write_text(json.dumps(out, indent=2) + "\n")
    print("guards:", json.dumps(guards))
    print("best c:", {k: v for k, v in out["best_c"].items() if not k.endswith("curve")},
          {k: {c: round(x, 3) for c, x in v.items()} for k, v in out["best_c"].items() if k.endswith("curve")})
    for n, s in out["slices"].items():
        print(f"  {n:18s} n={s['n']:3d} premium c1000 {s['premium_c1000']:+.3f}  tuned(c={s['best_c']}) "
              f"{s['premium_tuned']:+.3f} [{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]")
    for k, v in out["contrasts"].items():
        print(f"  {k}: {v['diff']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] (unpaired, query-level)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
