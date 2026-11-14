#!/usr/bin/env python3
"""Emit ablation/jobs.tsv (one row per arm x cell x repo) for ablation_bm25.slurm."""
from __future__ import annotations
from paths import DATA_ROOT
import argparse, sys
from pathlib import Path

ARMS = ("guard", "ablated", "control")
CELLS = {
    "A": ("qrels_cellA_ablated", "repos"),
    "B": ("qrels_cellB_ablated", "repos"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_ROOT)
    ap.add_argument("--cells", default="A")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.data / "ablation" / "jobs.tsv")

    rows = []
    for cell in args.cells.split(","):
        cell = cell.strip()
        qsub, clones = CELLS[cell]
        qdir = args.data / "ablation" / qsub
        if not qdir.is_dir():
            print(f"missing {qdir}", file=sys.stderr)
            return 1
        repos = sorted(p.stem for p in qdir.glob("*.jsonl")
                       if not p.stem.endswith("_audit"))
        for arm in ARMS:
            for repo in repos:
                rows.append("\t".join([
                    arm, cell, repo,
                    str(qdir / f"{repo}.jsonl"),
                    str(args.data / clones / repo),
                ]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n")
    print(f"wrote {out}: {len(rows)} tasks "
          f"(array 0-{len(rows) - 1}), arms={ARMS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
