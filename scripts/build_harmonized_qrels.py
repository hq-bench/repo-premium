"""Re-select the raw pools with the curated pool's rule applied to gold files.

The original raw selection applied the 1-3 file cap and the "exists at the
snapshot" check to every file a PR touched (changelog fragments, new tests,
docs). Curated gold never contains those, so the two pools were selected
under different rules. Here gold = indexed, non-test source files the change
edited; the cap and existence check apply to that set. `src/<pkg>/X` gold
paths are mapped to `<pkg>/X` when only the latter exists at the snapshot
(psf/requests moved to a src/ layout after the 2022 snapshot). Text filters
(`filter_reason`) are unchanged.

Writes <out>/<pool>/<org__repo>.jsonl with the runner's selection fields
(gold_files, n_gold_files, all_files_exist, filter_reason) rewritten, the
originals kept under `orig_*`, and <out>/summary.json with old/new counts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

PY_REPOS = {"astropy", "django", "matplotlib", "requests", "xarray", "pylint",
            "pytest", "scikit-learn", "sphinx", "sympy"}
TEST_RE = re.compile(r"(^|/)(tests?|testing|__tests__|testdata)(/|$)|(^|/)test_[^/]*$"
                     r"|_test\.py$|(^|/)conftest\.py$")


def tree(root: Path) -> set[str]:
    out = set()
    for d, dn, fs in os.walk(root):
        dn[:] = [x for x in dn if x != ".git"]
        for x in fs:
            out.add(os.path.relpath(os.path.join(d, x), root))
    return out


def harmonized_gold(touched: list[str], files: set[str]) -> list[str]:
    gold = []
    for g in touched:
        if not g.endswith(".py") or TEST_RE.search(g):
            continue
        if g not in files and g.startswith("src/") and g[4:] in files:
            g = g[4:]
        gold.append(g)
    return gold


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", action="append", required=True,
                   help="name:qrel_dir:snapshot_root (repeatable)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    summary = {}
    for spec in a.pool:
        name, qdir, snap = spec.split(":")
        (a.out / name).mkdir(parents=True, exist_ok=True)
        per_repo = {}
        for qf in sorted(Path(qdir).glob("*.jsonl")):
            repo = qf.stem
            if repo.startswith(".") or repo.split("__")[1] not in PY_REPOS:  # skip macOS ._ files
                continue
            files = tree(Path(snap) / repo)
            old = new = 0
            rows = []
            for line in open(qf):
                r = json.loads(line)
                if r.get("filter_reason"):
                    continue
                if r.get("all_files_exist") and 1 <= r["n_gold_files"] <= 3:
                    old += 1
                gold = harmonized_gold(r["gold_files"], files)
                if not (1 <= len(gold) <= 3 and all(g in files for g in gold)):
                    continue
                new += 1
                r = {**r,
                     "orig_gold_files": r["gold_files"],
                     "orig_n_gold_files": r["n_gold_files"],
                     "orig_all_files_exist": r.get("all_files_exist"),
                     "orig_selected": bool(r.get("all_files_exist") and 1 <= r["n_gold_files"] <= 3),
                     "gold_files": gold, "n_gold_files": len(gold),
                     "all_files_exist": True, "missing_files": []}
                rows.append(r)
            with open(a.out / name / qf.name, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            per_repo[repo] = {"old": old, "new": new}
        summary[name] = {"per_repo": per_repo,
                         "old": sum(v["old"] for v in per_repo.values()),
                         "new": sum(v["new"] for v in per_repo.values())}
        print(name, summary[name]["old"], "->", summary[name]["new"])
    json.dump(summary, open(a.out / "summary.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
