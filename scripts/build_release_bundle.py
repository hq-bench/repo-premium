#!/usr/bin/env python3
"""Package the CoREB-Repo benchmark cells for release.

The mining pipeline stores each cell's queries with the *raw* GitHub
`problem_statement`. Evaluation never sees that raw text -- it applies
`strip_pr_boilerplate` (smoke_test_django.py) first. A release that shipped
the stored field as-is would therefore both (a) hand reviewers a different
query than the one every number in the paper was computed on, and (b)
distribute PR-template and metadata blocks the paper's ethics statement says
are removed.

This script materialises the released form:

  1. `problem_statement` <- strip_pr_boilerplate(problem_statement), so the
     released text is exactly the evaluated text. The transform is idempotent,
     so downstream code that strips again is unaffected.
  2. `strip_pr_boilerplate_applied: true` marks the record as pre-stripped.
  3. `scrub_release_pii.redact` runs over every string, redacting personal
     email addresses that survive the boilerplate strip.

Then it writes MANIFEST.json with per-cell record counts, redaction counts,
and a sha256 per file, so the bundle can be checked after upload.

    python build_release_bundle.py --src ${REPO_PREMIUM_DATA:-data}/release_src \
                                   --out ${REPO_PREMIUM_DATA:-data}/release
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scrub_release_pii import redact  # noqa: E402
from run_retrieval import strip_pr_boilerplate  # noqa: E402

QUERY_FIELD = "problem_statement"


def scrub_strings(obj, seen):
    if isinstance(obj, str):
        return redact(obj, seen)
    if isinstance(obj, list):
        return [scrub_strings(v, seen) for v in obj]
    if isinstance(obj, dict):
        return {k: scrub_strings(v, seen) for k, v in obj.items()}
    return obj


def process_file(src: Path, dst: Path, seen: dict[str, int]) -> dict:
    n_rec = n_stripped = 0
    chars_before = chars_after = 0
    out_lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        n_rec += 1
        raw = rec.get(QUERY_FIELD)
        if isinstance(raw, str):
            stripped = strip_pr_boilerplate(raw)
            if stripped != raw:
                n_stripped += 1
            chars_before += len(raw)
            chars_after += len(stripped)
            rec[QUERY_FIELD] = stripped
            rec["strip_pr_boilerplate_applied"] = True
        rec = scrub_strings(rec, seen)
        out_lines.append(json.dumps(rec, ensure_ascii=False))
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(out_lines) + "\n"
    dst.write_text(payload, encoding="utf-8")
    return {
        "records": n_rec,
        "records_boilerplate_stripped": n_stripped,
        "query_chars_before": chars_before,
        "query_chars_after": chars_after,
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="directory holding the per-cell qrels* subdirectories")
    ap.add_argument("--out", type=Path, required=True,
                    help="destination directory for the release bundle")
    args = ap.parse_args(argv)

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    seen: dict[str, int] = {}
    manifest = {"cells": {}, "query_field": QUERY_FIELD,
                "strip_pr_boilerplate_applied": True}
    total_rec = 0
    for cell_dir in sorted(p for p in args.src.iterdir() if p.is_dir()):
        files = {}
        # `._*` are macOS AppleDouble resource forks left by an earlier
        # Mac-to-server copy; they are not data and must not ship.
        for f in sorted(f for f in cell_dir.glob("*.jsonl")
                        if not f.name.startswith("._")):
            files[f.name] = process_file(f, args.out / cell_dir.name / f.name, seen)
        for extra in sorted(f for f in cell_dir.glob("*.json")
                            if not f.name.startswith("._")):
            dst = args.out / cell_dir.name / extra.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(extra, dst)
        n = sum(v["records"] for v in files.values())
        total_rec += n
        manifest["cells"][cell_dir.name] = {"records": n, "files": files}
        print(f"{cell_dir.name:<26} {len(files):>3} file(s)  {n:>5} record(s)")

    manifest["total_records"] = total_rec
    manifest["pii_redactions"] = {
        "occurrences": sum(seen.values()),
        "distinct_addresses": len(seen),
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    print(f"\ntotal: {total_rec} records across {len(manifest['cells'])} cells")
    print(f"PII: redacted {sum(seen.values())} occurrence(s), "
          f"{len(seen)} distinct address(es)")
    print(f"wrote {args.out}/MANIFEST.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
