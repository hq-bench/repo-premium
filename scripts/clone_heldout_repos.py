"""clone_heldout_repos.py — Clone the 9 held-out repos for agent eval.

Reads scripts/heldout_repos.json and clones each
{org}/{repo} into {root}/{org}__{repo}/, pinned to HEAD at clone time.
Writes a manifest with the resolved SHAs for reproducibility.

Usage:
  python3 clone_heldout_repos.py \
      --manifest /path/to/heldout_repos.json \
      --root ${REPO_PREMIUM_DATA:-data}/repos_heldout \
      --out-manifest ${REPO_PREMIUM_DATA:-data}/repos_heldout/_manifest.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


def clone_one(org: str, repo: str, root: Path) -> dict:
    target = root / f"{org}__{repo}"
    if target.exists():
        # idempotent: just record the existing SHA
        sha = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        return {"org": org, "repo": repo, "sha": sha, "path": str(target), "status": "already_present"}
    url = f"https://github.com/{org}/{repo}.git"
    print(f"[clone] {url} -> {target}", file=sys.stderr)
    subprocess.check_call(
        ["git", "clone", "--quiet", "--depth", "200", url, str(target)]
    )
    sha = subprocess.check_output(
        ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
    ).strip()
    return {"org": org, "repo": repo, "sha": sha, "path": str(target), "status": "cloned"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    args = ap.parse_args()

    spec = json.loads(args.manifest.read_text())
    args.root.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for tier_key in ("tier1_in_distribution", "tier2_ood"):
        for entry in spec.get(tier_key, []):
            rec = clone_one(entry["org"], entry["repo"], args.root)
            rec["tier"] = tier_key
            rec["lang"] = entry.get("lang")
            rec["note"] = entry.get("note", "")
            records.append(rec)
            print(f"  -> {rec['org']}/{rec['repo']} @ {rec['sha'][:10]} ({rec['status']})",
                  file=sys.stderr)

    manifest = {
        "cloned_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "repos": records,
    }
    args.out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"[done] manifest at {args.out_manifest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
