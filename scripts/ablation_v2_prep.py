"""Lay out the fixed anchor-ablation runs for analyze_ablation.py, on source-file labels.

Inputs (shared FS):
  ablation_v2/dense/<slice>/<repo>/smoke_test_django.jsonl  probe runs: query ids <id>|ablated, <id>|control
  guard BM25: the original ablation runs (the guard query is the unchanged original query)
  canonical cache: the indexed-label mirror's canonical runs (already source-file labels)
Writes ablation_v2/analyze/<slice>/{res,den}/<arm>_cell<slice>/<repo>/smoke_test_django.jsonl with
gold sets relabeled as in build_source_only_mirror.py --indexed (tests and non-indexed files removed,
queries left without gold dropped), and a canon/ symlink.
"""
from paths import DATA_ROOT
import glob
import json
import os
from pathlib import Path

import build_source_only_mirror as M

W = DATA_ROOT / 'ablation_v2'
D = DATA_ROOT
CANON = (DATA_ROOT / 'smoke_test_results')
SLICES = {
    "B": {"exts": (".py",), "guard": lambda r: D / "ablation/results/guard_cellB" / r,
          "canon": CANON / "canon202607/jina05_cellB_c1000"},
    "rust": {"exts": (".rs",), "guard": lambda r: D / "ablation_r9/results/rust/guard" / r,
             "canon": CANON / "r9d_rust_c1000"},
    "ts": {"exts": (".ts", ".tsx", ".js", ".mjs", ".cjs"), "guard": lambda r: D / "ablation_r9/results/ts/guard" / r,
           "canon": CANON / "r9b_ts_c1000"},
}


def relabel(rows, exts):
    M.INDEXED, M.CTX_EXTS = True, exts
    stats = {"records": 0, "changed": 0, "test_files": 0, "dropped": 0}
    return [x for x in (M.fix(dict(r), stats) for r in rows) if x is not None], stats


def write(path: Path, rows):
    path.mkdir(parents=True, exist_ok=True)
    (path / "smoke_test_django.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def main():
    for sl, cfg in SLICES.items():
        root = W / "analyze" / sl
        n = {"ablated": 0, "control": 0, "guard": 0}
        for f in sorted(glob.glob(str(W / "dense" / sl / "*" / "smoke_test_django.jsonl"))):
            repo = f.split("/")[-2]
            probe = [json.loads(l) for l in open(f) if l.strip()]
            for arm in ("ablated", "control"):
                rows = [dict(r, query_id=r["query_id"].rsplit("|", 1)[0]) for r in probe
                        if r["query_id"].endswith("|" + arm)]
                rows, _ = relabel(rows, cfg["exts"])
                n[arm] += len(rows)
                write(root / "res" / f"{arm}_cell{sl}" / repo, rows)   # carries BM25 and dense
                write(root / "den" / f"{arm}_cell{sl}" / repo, rows)
            g = [json.loads(l) for l in open(cfg["guard"](repo) / "smoke_test_django.jsonl") if l.strip()]
            g, _ = relabel(g, cfg["exts"])
            n["guard"] += len(g)
            write(root / "res" / f"guard_cell{sl}" / repo, g)
        link = root / "canon"
        if link.is_symlink():
            link.unlink()
        os.symlink(cfg["canon"], link)
        print(sl, n)


if __name__ == "__main__":
    main()
