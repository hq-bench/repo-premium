"""Run rm3_bm25.run_repo for one (cell, repo): parallel driver for the harmonized raw pools."""
from paths import DATA_ROOT
import json
import sys
from pathlib import Path

import rm3_bm25 as M

cell, repo = sys.argv[1], sys.argv[2]
D = DATA_ROOT
root = D / ("repos_2024q1" if cell == "RA" else "repos")
dest = Path("out/rm3") / f"{cell}__{repo}.jsonl"
if not dest.is_file():
    rows = M.run_repo(Path(f"rm3root/qrels_cell{cell}_expanded/{repo}.jsonl"), root / repo)
    dest.write_text("".join(json.dumps(r) + "\n" for r in rows))
print("done", cell, repo, flush=True)
