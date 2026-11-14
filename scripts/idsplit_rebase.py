#!/usr/bin/env python3
"""Does identifier-splitting BM25 remove the remaining dense premium?

BM25's default tokenizer keeps `parse_config` / `parseConfig` as one token, so a query
that says "parse the config" cannot match it. This re-scores the premium against BM25
with identifier splitting (snake_case and camelCase -> subwords; smoke_test_django.py
--tokenize-camelcase) at the common chunk size (c=1000) and on whole files.

Pools: Cur-Pre, Raw-Post, Raw-Pre'22 (Python, local snapshot), Rust (tokio) and Go (hugo).
TypeScript already ran with splitting (r9b_ts, and its whole-file arm in the lockbox), so
it is reported from those runs. Raw-Pre'23 is omitted: it used a different snapshot.

Labels are source-only (test files removed, build_source_only_mirror.is_test), and nDCG@10
is re-scored from each run's stored top-10. Dense is the canonical jina-code-0.5b at c=1000.
CIs: repository-block bootstrap for the Python pools, query-level for single-repo pools.

Guards (printed, and the script exits 1 on failure):
  * unsplit whole-file Raw-Post rerun reproduces bm25_sweep202610 exactly;
  * unsplit c=1000 Go rerun reproduces r9c_go_c1000 exactly; Rust matches up to tie order.

Outputs ${REPO_PREMIUM_DATA:-data}/analysis/idsplit_rebase.json.
"""
from __future__ import annotations
from paths import DATA_ROOT

import json
import math
import random
import sys
from pathlib import Path

from build_source_only_mirror import is_test

SM = (DATA_ROOT / 'smoke_test_results')
ID = SM / "bm25_idsplit202610"
WHOLE = 1000000000
SEED, NBOOT = 20261010, 10000

POOLS = {
    "Cur-Pre": {"dense": SM / "canon202607/jina05_cellA_c1000",
                "plain_whole": SM / f"bm25_sweep202610/bm25_cellA_c{WHOLE}",
                "split_c1000": ID / "split_cellA_c1000", "split_whole": ID / f"split_cellA_c{WHOLE}"},
    "Raw-Post": {"dense": SM / "canon202607/jina05_cellB_c1000",
                 "plain_whole": SM / f"bm25_sweep202610/bm25_cellB_c{WHOLE}",
                 "split_c1000": ID / "split_cellB_c1000", "split_whole": ID / f"split_cellB_c{WHOLE}"},
    "Raw-Pre'22": {"dense": SM / "r8_2022_jina-0.5b_c1000",
                   "plain_whole": SM / f"gen_sweep202610/control_cellRB_c{WHOLE}",
                   "split_c1000": ID / "split_cellRB_c1000", "split_whole": ID / f"split_cellRB_c{WHOLE}"},
    "Rust": {"dense": SM / "r9d_rust_c1000", "plain_whole": ID / f"plain_r9_rs_c{WHOLE}",
             "split_c1000": ID / "split_r9_rs_c1000", "split_whole": ID / f"split_r9_rs_c{WHOLE}"},
    "Go": {"dense": SM / "r9c_go_c1000", "plain_whole": ID / f"plain_r9_go_c{WHOLE}",
           "split_c1000": ID / "split_r9_go_c1000", "split_whole": ID / f"split_r9_go_c{WHOLE}"},
}
SINGLE_REPO = {"Rust", "Go"}


def load(d: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(d.rglob("smoke_test_django.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["_repo"] = f.parent.name
                out[r["query_id"]] = r
    return out


def ndcg(top: list[str], gold: list[str]) -> float:
    g = set(gold)
    dcg = sum(1 / math.log2(i + 2) for i, f in enumerate(top[:10]) if f in g)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(len(g), 10)))


def ci(rows: list[tuple[str, float]], by_repo: bool) -> list[float]:
    rng, d = random.Random(SEED), []
    if by_repo:
        groups: dict[str, list[float]] = {}
        for rep, v in rows:
            groups.setdefault(rep, []).append(v)
        keys = sorted(groups)
        for _ in range(NBOOT):
            s = [v for _ in keys for v in groups[keys[rng.randrange(len(keys))]]]
            d.append(sum(s) / len(s))
    else:
        vals = [v for _, v in rows]
        for _ in range(NBOOT):
            s = [vals[rng.randrange(len(vals))] for _ in vals]
            d.append(sum(s) / len(s))
    d.sort()
    return [d[int(.025 * NBOOT)], d[int(.975 * NBOOT)]]


def guards() -> dict:
    out, ok = {}, True
    for name, new, old, exact in [
        ("Raw-Post plain whole", ID / f"guard_cellB_c{WHOLE}", SM / f"bm25_sweep202610/bm25_cellB_c{WHOLE}", True),
        ("Go plain c1000", ID / "guard_r9_go", SM / "r9c_go_c1000", True),
        ("Rust plain c1000", ID / "guard_r9_rs", SM / "r9d_rust_c1000", False),
    ]:
        a, b = load(new), load(old)
        common = [k for k in a if k in b]
        diffs = [abs(a[k]["bm25_ndcg10"] - b[k]["bm25_ndcg10"]) for k in common]
        out[name] = {"n_common": len(common), "max_abs_diff": max(diffs),
                     "mean_abs_diff": sum(diffs) / len(diffs)}
        if exact and max(diffs) > 1e-9:
            ok = False
        if not exact and out[name]["mean_abs_diff"] > 0.01:
            ok = False
    out["ok"] = ok
    return out


def main() -> int:
    res: dict = {"guards": guards(), "pools": {}}
    print(json.dumps(res["guards"], indent=1))
    for pool, p in POOLS.items():
        dense = load(p["dense"])
        bm = {"plain_c1000": dense}  # the dense run file also carries BM25 at c=1000
        bm.update({k: load(p[k]) for k in ("plain_whole", "split_c1000", "split_whole")})
        keys = [k for k in dense if all(k in bm[v] for v in bm)]
        rows_out = {"n": 0, "means": {}, "premium": {}}
        per: dict[str, list[tuple[str, float]]] = {v: [] for v in bm}
        dn: list[float] = []
        for k in keys:
            src = [g for g in dense[k]["gold_files"] if not is_test(g)]
            if not src:
                continue
            d = ndcg(dense[k]["dense_top10"], src)
            dn.append(d)
            for v in bm:
                per[v].append((dense[k]["_repo"], d - ndcg(bm[v][k]["bm25_top10"], src)))
        rows_out["n"] = len(dn)
        rows_out["means"]["dense"] = sum(dn) / len(dn)
        for v, rows in per.items():
            prem = sum(x for _, x in rows) / len(rows)
            rows_out["means"][f"bm25_{v}"] = rows_out["means"]["dense"] - prem
            rows_out["premium"][v] = {"mean": prem, "ci95": ci(rows, pool not in SINGLE_REPO)}
        best = min(rows_out["premium"], key=lambda v: rows_out["premium"][v]["mean"])
        rows_out["best_bm25"] = best
        res["pools"][pool] = rows_out
        f = lambda v: f"{rows_out['premium'][v]['mean']:+.3f} [{rows_out['premium'][v]['ci95'][0]:+.3f},{rows_out['premium'][v]['ci95'][1]:+.3f}]"
        print(f"{pool:11s} n={rows_out['n']:3d} dense={rows_out['means']['dense']:.3f} | "
              + " | ".join(f"{v} {f(v)}" for v in per) + f" | best={best}")
    (DATA_ROOT / 'analysis/idsplit_rebase.json').write_text(json.dumps(res, indent=2) + "\n")
    return 0 if res["guards"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
