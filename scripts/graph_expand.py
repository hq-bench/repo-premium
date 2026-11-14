"""R6 Step 2 — graph neighborhood expansion.

Loads a `graph.jsonl` produced by `build_code_graph.py` and exposes
`neighborhood(file, depth)` returning the set of files reachable via:

  * outgoing calls   (this file calls into them; resolved_file)
  * incoming calls   (they call into this file; reverse index)
  * outgoing inherit (this file's classes inherit from them)
  * incoming inherit (their classes inherit from this file)
  * same directory   (siblings — optional, controlled by include_siblings)

Usage as a library:
    from graph_expand import GraphIndex
    g = GraphIndex.from_jsonl(Path(".../graph.jsonl"))
    nbrs = g.neighborhood("django/db/models/query.py", depth=1)

Usage as a CLI (debug):
    python scripts/graph_expand.py \
        --graph-jsonl ${REPO_PREMIUM_DATA:-data}/graph/django__django/graph.jsonl \
        --seed-file django/db/models/query.py \
        --depth 1
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GraphIndex:
    # Forward (file → set of files it depends on via calls / inherits).
    calls_out: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    inherits_out: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # Reverse (file → set of files that depend on it).
    calls_in: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    inherits_in: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # Index of every file we've seen.
    files: set[str] = field(default_factory=set)

    @classmethod
    def from_jsonl(cls, path: Path) -> "GraphIndex":
        g = cls()
        with path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                src = rec["file"]
                g.files.add(src)
                for c in rec.get("calls", []):
                    tgt = c.get("resolved_file")
                    if tgt and tgt != src:
                        g.calls_out[src].add(tgt)
                        g.calls_in[tgt].add(src)
                for inh in rec.get("inherits", []):
                    tgt = inh.get("resolved_file")
                    if tgt and tgt != src:
                        g.inherits_out[src].add(tgt)
                        g.inherits_in[tgt].add(src)
        return g

    def neighborhood(
        self,
        seed: str,
        depth: int = 1,
        include_siblings: bool = False,
    ) -> set[str]:
        """BFS over the union of {calls_out, calls_in, inherits_out, inherits_in} up to depth.

        Returns the set of reached files *excluding* the seed itself.
        """
        if seed not in self.files:
            return set()
        visited: set[str] = {seed}
        frontier: set[str] = {seed}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for f in frontier:
                next_frontier |= self.calls_out.get(f, set())
                next_frontier |= self.calls_in.get(f, set())
                next_frontier |= self.inherits_out.get(f, set())
                next_frontier |= self.inherits_in.get(f, set())
            next_frontier -= visited
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        if include_siblings:
            sib = self._siblings(seed)
            visited |= sib
        visited.discard(seed)
        return visited

    def _siblings(self, seed: str) -> set[str]:
        d = str(Path(seed).parent)
        return {f for f in self.files if str(Path(f).parent) == d and f != seed}

    def degree_stats(self) -> dict[str, float]:
        n = max(len(self.files), 1)
        return {
            "n_files": len(self.files),
            "avg_calls_out": sum(len(v) for v in self.calls_out.values()) / n,
            "avg_calls_in": sum(len(v) for v in self.calls_in.values()) / n,
            "avg_inherits_out": sum(len(v) for v in self.inherits_out.values()) / n,
            "avg_inherits_in": sum(len(v) for v in self.inherits_in.values()) / n,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-jsonl", type=Path, required=True)
    ap.add_argument("--seed-file", type=str, required=True,
                    help="Repo-relative path of the seed file.")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--include-siblings", action="store_true")
    args = ap.parse_args()

    g = GraphIndex.from_jsonl(args.graph_jsonl)
    print(f"[graph] loaded {len(g.files)} files from {args.graph_jsonl}")
    stats = g.degree_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<22s} = {v:.2f}")
        else:
            print(f"  {k:<22s} = {v}")

    nbrs = g.neighborhood(args.seed_file, depth=args.depth,
                          include_siblings=args.include_siblings)
    print(f"\n[graph] neighborhood of {args.seed_file!r} "
          f"at depth={args.depth} siblings={args.include_siblings}: "
          f"{len(nbrs)} files")
    for f in sorted(nbrs)[:30]:
        print(f"  {f}")
    if len(nbrs) > 30:
        print(f"  ... and {len(nbrs) - 30} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
