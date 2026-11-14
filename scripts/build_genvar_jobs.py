#!/usr/bin/env python3
"""Emit genarm_genvar/jobs.tsv for genvar_bm25.slurm.

Generator-variance scoring: the `control` arm (original query) is
model-independent and already exists from the original genarm run, so only
`expanded` and `generated` are re-scored, once per additional generator.
Columns: MODEL ARM CELL REPO QREL_FILE CLONE_DIR
"""
from __future__ import annotations
from paths import DATA_ROOT
import argparse, sys
from pathlib import Path

ARMS = ("expanded", "generated")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=DATA_ROOT)
    ap.add_argument("--models", default="gemma-4-31B-it,deepseek-v4-flash")
    ap.add_argument("--cells", default="A,B")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.data / "genarm_genvar"
    out = args.out or (root / "jobs.tsv")

    rows = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        for cell in [c.strip() for c in args.cells.split(",") if c.strip()]:
            qdir = root / model / f"qrels_cell{cell}_expanded"
            if not qdir.is_dir():
                print(f"missing {qdir}", file=sys.stderr)
                return 1
            repos = sorted(p.stem for p in qdir.glob("*.jsonl"))
            for arm in ARMS:
                for repo in repos:
                    rows.append("\t".join([
                        model, arm, cell, repo,
                        str(qdir / f"{repo}.jsonl"),
                        str(args.data / "repos" / repo),
                    ]))
    out.write_text("\n".join(rows) + "\n")
    print(f"wrote {out}: {len(rows)} tasks (array 0-{len(rows)-1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
