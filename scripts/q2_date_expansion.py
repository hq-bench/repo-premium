#!/usr/bin/env python3
"""Q2 date contrast at repository level: raw PR queries before vs after encoder cutoffs.

Each repository contributes a pre-cutoff pool and a post-cutoff pool mined with the same
raw-PR pipeline against ONE pinned corpus snapshot, so the contrast varies the vintage
alone. Sources:

  new (11 repos)       2022 PRs vs >= 2026-01-01 PRs, snapshots pinned at the last commit
                       before 2023-07-01 (q2_prepare.sh); q2_{pre,post}_* result dirs
  existing (10 repos)  Raw-Pre'22 (r8_2022) vs Raw-Post (Cell B), both on the repos/ snapshot

Date effect = mean premium(post) - mean premium(pre), premium = dense(jina-code-0.5b, c=1000)
minus BM25, with BM25 at c=1000 (common protocol) and over whole files (tuned).
Repository-block bootstrap: resample repositories, each bringing both of its pools
(10,000 resamples, seed 0). Repositories with an empty pool are skipped for that analysis.
"""
from __future__ import annotations
from paths import DATA_ROOT

import json
import random
from collections import defaultdict
from pathlib import Path

SM = (DATA_ROOT / 'smoke_test_results')
SOURCES = {
    "new": {"pre": (SM / "q2_pre_jina05_c1000", SM / "q2_pre_bm25_whole"),
            "post": (SM / "q2_post_jina05_c1000", SM / "q2_post_bm25_whole")},
    "existing": {"pre": (SM / "r8_2022_jina-0.5b_c1000", SM / "gen_sweep202610/control_cellRB_c1000000000"),
                 "post": (SM / "canon202607/jina05_cellB_c1000", SM / "bm25_sweep202610/bm25_cellB_c1000000000")},
}


def load(d: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for f in sorted(d.glob("*/smoke_test_django.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[f.parent.name][r["query_id"]] = r
    return out


def premiums(src: dict, tuned: bool) -> dict[str, dict[str, list[float]]]:
    """repo -> pool -> per-query premiums."""
    res: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for pool, (ddir, wdir) in src.items():
        dense, whole = load(ddir), load(wdir)
        for repo, rows in dense.items():
            vals = []
            for qid, r in rows.items():
                b = whole[repo][qid]["bm25_ndcg10"] if tuned else r["bm25_ndcg10"]
                vals.append(r["dense_ndcg10"] - b)
            res[repo][pool] = vals
    return {r: v for r, v in res.items() if v.get("pre") and v.get("post")}


def effect(repos: dict, keys: list[str]) -> float:
    pre = [x for k in keys for x in repos[k]["pre"]]
    post = [x for k in keys for x in repos[k]["post"]]
    return sum(post) / len(post) - sum(pre) / len(pre)


def boot(repos: dict, n: int = 10000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    keys = sorted(repos)
    d = sorted(effect(repos, [keys[rng.randrange(len(keys))] for _ in keys]) for _ in range(n))
    return [d[int(.025 * n)], d[int(.975 * n)]]


def main() -> int:
    out = {}
    for tuned in (False, True):
        tag = "bm25_whole" if tuned else "bm25_c1000"
        per = {name: premiums(src, tuned) for name, src in SOURCES.items()}
        both = {**{f"new:{k}": v for k, v in per["new"].items()},
                **{f"existing:{k}": v for k, v in per["existing"].items()}}
        res = {}
        for name, repos in (("new", per["new"]), ("existing", per["existing"]), ("pooled", both)):
            res[name] = {"n_repos": len(repos),
                         "n_pre": sum(len(v["pre"]) for v in repos.values()),
                         "n_post": sum(len(v["post"]) for v in repos.values()),
                         "date_effect": effect(repos, sorted(repos)), "ci95": boot(repos),
                         "per_repo": {k: {"pre": sum(v["pre"]) / len(v["pre"]), "post": sum(v["post"]) / len(v["post"]),
                                          "n_pre": len(v["pre"]), "n_post": len(v["post"])} for k, v in repos.items()}}
        out[tag] = res
        print(f"== {tag}")
        for name, r in res.items():
            print(f"  {name:8s} repos={r['n_repos']:2d} pre n={r['n_pre']:3d} post n={r['n_post']:3d} "
                  f"date effect {r['date_effect']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]")
    (DATA_ROOT / 'analysis/q2_date_expansion.json').write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
