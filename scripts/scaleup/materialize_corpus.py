#!/usr/bin/env python3
"""Materialize a task's language-filtered corpus from a bare mirror.

Runs inside the retrieval container (compute nodes have no git) and writes the
exact file set the audit enumerated.

`git archive` cannot be used for this. It honours `.gitattributes`
`export-ignore` while `git ls-tree` does not, so the two disagree on
repositories that export-ignore source paths -- pandas excludes `setup.py` and
`pandas/tests/io/parser/data`, a 93-file gap on a 1,546-file corpus. Left
unchecked that silently shrinks the corpus, and would silently drop a gold file
that happened to be export-ignored.

This tool therefore enumerates with `ls-tree` (identical to `audit_corpora.py`)
and extracts blob contents through a single long-lived `git cat-file --batch`,
so the materialized corpus equals the audited corpus by construction rather
than by coincidence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BLOB_MODES = {b"100644", b"100755"}


def read_exactly(stream, n: int) -> bytes:
    """cat-file --batch payloads must be consumed by length, not by newline."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        block = stream.read(remaining)
        if not block:
            raise RuntimeError(f"unexpected EOF with {remaining} bytes outstanding")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--exts", required=True,
                        help="Comma-separated extensions, e.g. '.py,.pyi'.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expect-files", type=int, default=-1,
                        help="Audited file count; mismatch fails closed.")
    args = parser.parse_args()

    exts = tuple(
        e.strip() if e.strip().startswith(".") else "." + e.strip()
        for e in args.exts.split(",") if e.strip()
    )
    if not exts:
        print("no extensions given", file=sys.stderr)
        return 1

    listing = subprocess.run(
        ["git", "-C", args.mirror, "ls-tree", "-r", "-z", args.commit],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if listing.returncode != 0:
        print(f"ls-tree failed: {listing.stderr.decode(errors='replace')[:400]}",
              file=sys.stderr)
        return 1

    entries: list[tuple[str, str]] = []
    for record in listing.stdout.split(b"\0"):
        if not record:
            continue
        meta, _, raw_path = record.partition(b"\t")
        if not raw_path:
            continue
        parts = meta.split(b" ")
        if len(parts) < 3 or parts[0] not in BLOB_MODES:
            continue
        path = raw_path.decode("utf-8", errors="surrogateescape")
        if path.endswith(exts):
            entries.append((parts[2].decode(), path))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["git", "-C", args.mirror, "cat-file", "--batch"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.stdin and proc.stdout
    written = 0
    try:
        for oid, path in entries:
            proc.stdin.write((oid + "\n").encode())
            proc.stdin.flush()
            header = proc.stdout.readline().split()
            if len(header) < 3 or header[1] != b"blob":
                raise RuntimeError(f"unexpected cat-file header for {path}: {header!r}")
            payload = read_exactly(proc.stdout, int(header[2]))
            proc.stdout.read(1)  # trailing newline after the payload
            target = args.out_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            written += 1
    finally:
        proc.stdin.close()
        proc.wait()

    if args.expect_files >= 0 and written != args.expect_files:
        print(f"corpus mismatch: wrote {written}, audited {args.expect_files}",
              file=sys.stderr)
        return 1
    print(f"materialized {written} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
