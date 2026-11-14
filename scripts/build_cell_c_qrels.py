"""Split Cell C records into per-repo qrel files for the standard runners.

Reads ${REPO_PREMIUM_DATA:-data}/coreb_repo_cell_c.jsonl and emits per-repo JSONL files in the same
schema as ${REPO_PREMIUM_DATA:-data}/coreb_repo_qrels/qrels/*.jsonl so the existing
smoke_test_django.py / graph_search.py / graph_rerank.py pipelines can ingest
them unchanged.

Per-record schema written:
  {query_id, repo, problem_statement, gold_files, base_commit,
   source_query_id, cell_origin}

Out dir layout:
  <out>/
    django__django.jsonl
    matplotlib__matplotlib.jsonl
    ...

Usage:
  python scripts/build_cell_c_qrels.py \
      --src ${REPO_PREMIUM_DATA:-data}/coreb_repo_cell_c.jsonl \
      --out-dir ${REPO_PREMIUM_DATA:-data}/coreb_repo_qrels_cellC
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
from collections import defaultdict
from pathlib import Path


def repo_to_filename(repo: str) -> str:
    """`django/django` -> `django__django`."""
    return repo.replace("/", "__")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path,
                    default=(DATA_ROOT / 'coreb_repo_cell_c.jsonl'))
    ap.add_argument("--out-dir", type=Path,
                    default=(DATA_ROOT / 'coreb_repo_qrels_cellC'))
    ap.add_argument("--require-accepted", action="store_true", default=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_repo: dict[str, list[dict]] = defaultdict(list)
    n_total = 0
    n_skipped = 0
    for line in args.src.open():
        r = json.loads(line)
        if args.require_accepted and not r.get("accepted", False):
            n_skipped += 1
            continue
        repo = r["repo"]
        rec = {
            "query_id": r["query_id"],
            "repo": repo,
            "problem_statement": r["problem_statement"],
            "gold_files": r["gold_files"],
            "base_commit": r["base_commit"],
            # Fields required by smoke_test_django.load_qrels() filter.
            "n_gold_files": len(r["gold_files"]),
            "all_files_exist": True,
            "missing_files": [],
            # Extra provenance fields (ignored by runners, useful for aggregation)
            "source_query_id": r["source_query_id"],
            "cell_origin": r["cell_origin"],
            "n_identifiers_rewritten": r["n_identifiers_rewritten"],
        }
        by_repo[repo].append(rec)
        n_total += 1

    summary = {}
    for repo, recs in sorted(by_repo.items()):
        fname = args.out_dir / f"{repo_to_filename(repo)}.jsonl"
        with fname.open("w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        summary[repo_to_filename(repo)] = len(recs)
        print(f"  {fname.name}: {len(recs)}")

    (args.out_dir / "qrels_summary.json").write_text(
        json.dumps({
            "total_queries": n_total,
            "skipped_not_accepted": n_skipped,
            "by_repo": summary,
        }, indent=2)
    )
    print(f"\ntotal: {n_total} queries across {len(by_repo)} repos")
    print(f"skipped (not accepted): {n_skipped}")
    print(f"wrote {args.out_dir}/qrels_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
