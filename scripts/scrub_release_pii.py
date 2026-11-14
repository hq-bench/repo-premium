#!/usr/bin/env python3
"""Release-time PII scrub for CoREB-Repo query/qrel JSONL files.

Redacts email addresses from every string field of each record. Run over
the packaged benchmark cells before uploading supplementary material:

    python scrub_release_pii.py --dry-run release_dir/*.jsonl   # report only
    python scrub_release_pii.py release_dir/*.jsonl             # rewrite

Files are rewritten in place; a `.pre_scrub.bak` copy is kept next to each
input the first time it is modified. Exit code is the number of files that
contained at least one redaction, so `&& echo clean` only fires on 0.

Why the match is filtered rather than raw
-----------------------------------------
A bare `user@host.tld` regex is far too greedy on code-bearing query text,
and over-redaction is not a safe default here: the tokens it eats are rare
identifiers, which are precisely the lexical anchors BM25 scores on. Left
unfiltered it destroys, in this corpus,

  * TypeScript test filenames carrying an npm scope --
    `npm-install-@types-works.js` (42 occurrences across the TS cells);
  * npm compare tokens -- `knip@6.16.0...knip`, `webpack-cli@5.0.0...webpack`;
  * SSH key-type constants -- `sk-ssh-ed25519@openssh.com`.

None is personal data, and all three are the kind of token the benchmark
exists to measure. So a match is redacted only if it survives the filters
below. RFC 2606 reserved domains (`example.com`, and friends) are exempt for
the same reason: they cannot identify anyone, so redacting them would edit
query text for no privacy gain.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
REDACTION = "[email-redacted]"

# RFC 2606 / RFC 6761 reserved names: cannot resolve to a real mailbox.
RESERVED_DOMAINS = ("example.com", "example.org", "example.net",
                    "example.edu")
RESERVED_TLDS = (".example", ".invalid", ".test", ".localhost")

# A "domain" ending in one of these is a filename, not a host.
FILE_EXTS = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".py", ".java",
             ".go", ".rs", ".rb", ".php", ".c", ".cc", ".cpp", ".h", ".hpp",
             ".cs", ".kt", ".swift", ".scala", ".sh", ".bash", ".zsh",
             ".md", ".rst", ".txt", ".json", ".yml", ".yaml", ".toml",
             ".ini", ".cfg", ".xml", ".html", ".htm", ".css", ".scss",
             ".sql", ".lock", ".snap", ".map")

# Technical constants that are email-shaped but name a protocol/service,
# not a person.
TECHNICAL_LITERALS = {
    "git@github.com",
    "sk-ssh-ed25519@openssh.com",
    "sk-ecdsa-sha2-nistp256@openssh.com",
    "sk-ssh-ed25519-cert-v01@openssh.com",
    "sk-ecdsa-sha2-nistp256-cert-v01@openssh.com",
}


def is_personal(match: str) -> bool:
    """True if `match` plausibly identifies a real mailbox worth redacting."""
    if match.lower() in TECHNICAL_LITERALS:
        return False
    local, _, domain = match.rpartition("@")
    dom = domain.lower()
    # `pkg@1.2.3...pkg` style compare tokens: no real host has a run of dots.
    if ".." in dom:
        return False
    if dom.endswith(FILE_EXTS):
        return False
    if dom in RESERVED_DOMAINS or dom.endswith(RESERVED_TLDS):
        return False
    # A host label must contain at least one letter; `6.16.0` must not pass.
    if not any(c.isalpha() for c in dom.rsplit(".", 1)[0]):
        return False
    return True


def redact(text: str, seen: dict[str, int] | None = None) -> str:
    def sub(m: re.Match) -> str:
        if not is_personal(m.group(0)):
            return m.group(0)
        if seen is not None:
            seen[m.group(0)] = seen.get(m.group(0), 0) + 1
        return REDACTION
    return EMAIL_RE.sub(sub, text)


def scrub(obj, seen=None):
    """Recursively redact emails in every string of a JSON-compatible object."""
    if isinstance(obj, str):
        return redact(obj, seen)
    if isinstance(obj, list):
        return [scrub(v, seen) for v in obj]
    if isinstance(obj, dict):
        return {k: scrub(v, seen) for k, v in obj.items()}
    return obj


def scrub_file(path: Path, seen: dict[str, int], dry_run: bool = False) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    out, n_hits = [], 0
    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        record = json.loads(line)
        scrubbed = scrub(record, seen)
        if scrubbed != record:
            n_hits += 1
        out.append(json.dumps(scrubbed, ensure_ascii=False))
    if n_hits and not dry_run:
        bak = path.with_suffix(path.suffix + ".pre_scrub.bak")
        if not bak.exists():
            shutil.copy2(path, bak)
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return n_hits


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="JSONL files to scrub")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be redacted without rewriting")
    ap.add_argument("--quiet", action="store_true",
                    help="only print the per-run summary")
    args = ap.parse_args(argv)
    if not args.files:
        ap.print_help()
        return 0

    seen: dict[str, int] = {}
    dirty = 0
    for path in args.files:
        hits = scrub_file(path, seen, dry_run=args.dry_run)
        if hits and not args.quiet:
            print(f"{path}: {hits} record(s) redacted")
        dirty += bool(hits)
    verb = "would redact" if args.dry_run else "redacted"
    print(f"\n{verb} {sum(seen.values())} address occurrence(s), "
          f"{len(seen)} distinct, across {dirty} file(s)")
    for addr, n in sorted(seen.items(), key=lambda kv: -kv[1]):
        local, _, dom = addr.partition("@")
        print(f"  {n:>4}  {local[:2]}***@{dom}")
    return dirty


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
