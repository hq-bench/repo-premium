"""sample_heldout_eval_queries.py — Stratified sample of held-out qrels for agent eval.

Reads {qrels_dir}/*.jsonl (one file per repo), keeps records where
`all_files_exist=True` and `filter_reason` is null, then takes a fixed quota
per repo to produce a balanced 30-query stratified sample.

Quotas are hardcoded to match the 238-usable pool distribution we mined
on 2026-06-28. Edit `QUOTAS` to rescale.

Usage:
  python3 sample_heldout_eval_queries.py \
      --qrels-dir ${REPO_PREMIUM_DATA:-data}/qrels_heldout \
      --repos-root ${REPO_PREMIUM_DATA:-data}/repos_heldout \
      --out ${REPO_PREMIUM_DATA:-data}/agent_eval_30.jsonl \
      --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

QUOTAS = {
    "encode__httpx":            2,
    "fastapi__typer":           3,
    "pallets__flask":           3,
    "aio-libs__aiohttp":        3,
    "tornadoweb__tornado":      4,
    "pandas-dev__pandas":       4,
    "pydantic__pydantic":       4,
    "python-pillow__Pillow":    3,
    "python-poetry__poetry":    4,
}


def load_usable(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("all_files_exist") and not r.get("filter_reason"):
            out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels-dir", type=Path, required=True)
    ap.add_argument("--repos-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=0,
                    help="Total sample size. 0 uses the hardcoded QUOTAS (n=30). "
                         "Otherwise quotas are rescaled proportionally to each "
                         "repository's usable pool, by largest remainder.")
    ap.add_argument("--include", type=Path, default=None,
                    help="An existing sample to nest inside this one, so a larger "
                         "run is a superset of the smaller and the two are "
                         "directly comparable.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    keep: set[str] = set()
    if args.include:
        keep = {json.loads(l)["query_id"]
                for l in args.include.read_text().splitlines() if l.strip()}
        print(f"[nest] carrying {len(keep)} queries from {args.include.name}")
    quotas = dict(QUOTAS)
    if args.n:
        pools = {}
        for repo_dir_name in QUOTAS:
            pools[repo_dir_name] = len(load_usable(args.qrels_dir / f"{repo_dir_name}.jsonl"))
        total = sum(pools.values())
        exact = {r: args.n * pools[r] / total for r in pools}
        quotas = {r: min(int(exact[r]), pools[r]) for r in pools}
        # largest-remainder top-up, never exceeding a repository's pool
        while sum(quotas.values()) < min(args.n, total):
            r = max((r for r in pools if quotas[r] < pools[r]),
                    key=lambda r: exact[r] - quotas[r], default=None)
            if r is None:
                break
            quotas[r] += 1
        print(f"[quota] n={sum(quotas.values())} across {len(quotas)} repos: {quotas}")
    sample: list[dict] = []
    for repo_dir_name, quota in quotas.items():
        qrel_path = args.qrels_dir / f"{repo_dir_name}.jsonl"
        usable = load_usable(qrel_path)
        if len(usable) < quota:
            raise SystemExit(
                f"Repo {repo_dir_name}: pool {len(usable)} < quota {quota}"
            )
        must = [r for r in usable if r["query_id"] in keep]
        rest = [r for r in usable if r["query_id"] not in keep]
        if len(must) >= quota:
            picked = must[:quota]
        else:
            picked = must + rng.sample(rest, quota - len(must))
        for rec in picked:
            rec["repo_dir"] = str(args.repos_root / repo_dir_name)
            sample.append(rec)
        print(f"  {repo_dir_name:<32} pool={len(usable):>3}  picked={quota}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for rec in sample:
            f.write(json.dumps(rec) + "\n")
    print(f"[done] {len(sample)} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
