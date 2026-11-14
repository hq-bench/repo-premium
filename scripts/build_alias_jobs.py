#!/usr/bin/env python3
"""Emit ablation/jobs_alias.tsv for the review-#14 alias arm.

Same row format as build_ablation_jobs.py (arm, cell, repo, qrel, clone), so
ablation_bm25.slurm / ablation_dense.slurm run it unchanged with
`--export=ALL,CONFIG=.../jobs_alias.tsv`. Only the `alias` arm is emitted:
guard, ablated and control are already scored and are reused as-is.
"""
from __future__ import annotations
from paths import DATA_ROOT
import argparse, sys
from pathlib import Path

CELLS = {"A": "qrels_cellA_alias", "B": "qrels_cellB_alias"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_ROOT)
    ap.add_argument("--cells", default="A,B")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.data / "ablation" / "jobs_alias.tsv")

    rows = []
    for cell in (c.strip() for c in args.cells.split(",")):
        qdir = args.data / "ablation" / CELLS[cell]
        if not qdir.is_dir():
            print(f"missing {qdir} -- run anchor_alias_control.py first", file=sys.stderr)
            return 1
        repos = sorted(p.stem for p in qdir.glob("*.jsonl")
                       if not p.stem.endswith(("_audit", "_llm_cache")))
        for repo in repos:
            rows.append("\t".join(["alias", cell, repo, str(qdir / f"{repo}.jsonl"),
                                   str(args.data / "repos" / repo)]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n")
    print(f"wrote {out}: {len(rows)} tasks (array 0-{len(rows) - 1})")
    for i, r in enumerate(rows):
        print(i, *r.split("\t")[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
