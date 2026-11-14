"""R6 scaffold — build a Python symbol graph for one repo.

For each .py file under --clone-dir:
  * extract function defs, class defs, imports, class inheritance, call sites.
  * resolve calls / inheritance to the defining file when possible (intra-repo).
  * write a graph JSONL with nodes + edges.

Output (--out-dir / "graph.jsonl"):
  one record per file:
    {
      "file": "django/db/models/query.py",
      "n_lines": 2543,
      "defs":   [{"kind": "function|class", "name": "...", "line": 12, "qualname": "..."}, ...],
      "imports": [{"module": "django.db.models", "name": "Manager", "alias": null, "line": 4}, ...],
      "inherits": [{"class": "Foo", "parent": "Bar", "line": 50, "resolved_file": "django/db/models/base.py" | null}, ...],
      "calls":   [{"callee_name": "filter", "line": 102, "resolved_file": "django/db/models/manager.py" | null}, ...],
    }

This is purely intra-repo static analysis. No call graph beyond simple
name-resolution. Dynamic dispatch (ORM metaclasses, getattr) is by definition
out of reach for static AST; we record coverage stats so we can quantify how
much the graph layer can theoretically pull in.

Usage:
  python scripts/build_code_graph.py \
      --clone-dir /path/to/django__django \
      --out-dir   ${REPO_PREMIUM_DATA:-data}/coreb_repo_graph/django__django
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def iter_py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py"))


def parse_file(path: Path) -> ast.AST | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return ast.parse(text, filename=str(path))
    except (SyntaxError, ValueError):
        return None


def _qualname(stack: list[str], name: str) -> str:
    return ".".join(stack + [name]) if stack else name


def extract_symbols(tree: ast.AST) -> dict[str, list[dict]]:
    """Walk an AST and emit defs / imports / inherits / calls records.

    Calls and inherits are recorded by *name only* — resolution happens later
    once we have a repo-wide symbol index.
    """
    defs: list[dict] = []
    imports: list[dict] = []
    inherits: list[dict] = []
    calls: list[dict] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            defs.append({
                "kind": "function",
                "name": node.name,
                "qualname": _qualname(self.scope, node.name),
                "line": node.lineno,
            })
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            defs.append({
                "kind": "class",
                "name": node.name,
                "qualname": _qualname(self.scope, node.name),
                "line": node.lineno,
            })
            for base in node.bases:
                pname = None
                if isinstance(base, ast.Name):
                    pname = base.id
                elif isinstance(base, ast.Attribute):
                    pname = base.attr
                if pname:
                    inherits.append({
                        "class": node.name,
                        "parent": pname,
                        "line": node.lineno,
                    })
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                imports.append({
                    "module": alias.name,
                    "name": alias.name.split(".")[-1],
                    "alias": alias.asname,
                    "line": node.lineno,
                })

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module = node.module or ""
            for alias in node.names:
                imports.append({
                    "module": module,
                    "name": alias.name,
                    "alias": alias.asname,
                    "line": node.lineno,
                })

        def visit_Call(self, node: ast.Call) -> None:
            callee_name = None
            if isinstance(node.func, ast.Name):
                callee_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee_name = node.func.attr
            if callee_name:
                calls.append({
                    "callee_name": callee_name,
                    "line": getattr(node, "lineno", -1),
                })
            self.generic_visit(node)

    Visitor().visit(tree)
    return {"defs": defs, "imports": imports, "inherits": inherits, "calls": calls}


def build_def_index(per_file: dict[str, dict]) -> dict[str, list[str]]:
    """name -> list of files that define a top-level (or class-level) symbol with that name."""
    idx: dict[str, list[str]] = defaultdict(list)
    for fp, rec in per_file.items():
        seen: set[str] = set()
        for d in rec["defs"]:
            if d["name"] not in seen:
                idx[d["name"]].append(fp)
                seen.add(d["name"])
    return idx


def resolve_names(per_file: dict[str, dict], def_idx: dict[str, list[str]]) -> dict[str, Any]:
    """For each call / inherit, resolve to a unique defining file when name is unambiguous.

    Reports coverage stats: how many edges are resolvable.
    """
    resolved_calls = 0
    total_calls = 0
    resolved_inherits = 0
    total_inherits = 0
    for fp, rec in per_file.items():
        for c in rec["calls"]:
            total_calls += 1
            files = def_idx.get(c["callee_name"], [])
            if len(files) == 1:
                c["resolved_file"] = files[0]
                resolved_calls += 1
            elif len(files) > 1:
                c["resolved_file"] = None
                c["candidates"] = files[:5]
            else:
                c["resolved_file"] = None
        for inh in rec["inherits"]:
            total_inherits += 1
            files = def_idx.get(inh["parent"], [])
            if len(files) == 1:
                inh["resolved_file"] = files[0]
                resolved_inherits += 1
            elif len(files) > 1:
                inh["resolved_file"] = None
                inh["candidates"] = files[:5]
            else:
                inh["resolved_file"] = None
    return {
        "calls_total": total_calls,
        "calls_resolved_unique": resolved_calls,
        "calls_resolution_rate": resolved_calls / max(total_calls, 1),
        "inherits_total": total_inherits,
        "inherits_resolved_unique": resolved_inherits,
        "inherits_resolution_rate": resolved_inherits / max(total_inherits, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clone-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-files", type=int, default=0,
                    help="If >0, only parse first N files (for quick smoke tests).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[graph] listing .py files under {args.clone_dir}", file=sys.stderr)
    py_files = iter_py_files(args.clone_dir)
    if args.max_files > 0:
        py_files = py_files[: args.max_files]
    print(f"[graph] {len(py_files)} files", file=sys.stderr)

    t0 = time.time()
    per_file: dict[str, dict] = {}
    n_parsed = 0
    n_failed = 0
    for fp in py_files:
        tree = parse_file(fp)
        if tree is None:
            n_failed += 1
            continue
        rel = fp.relative_to(args.clone_dir).as_posix()
        try:
            n_lines = fp.read_text(encoding="utf-8", errors="ignore").count("\n") + 1
        except Exception:
            n_lines = 0
        rec = extract_symbols(tree)
        rec["file"] = rel
        rec["n_lines"] = n_lines
        per_file[rel] = rec
        n_parsed += 1
    dt = time.time() - t0
    print(f"[graph] parsed {n_parsed} files, {n_failed} failed in {dt:.1f}s", file=sys.stderr)

    print(f"[graph] indexing top-level symbols ...", file=sys.stderr)
    def_idx = build_def_index(per_file)
    print(f"[graph]   {len(def_idx)} distinct symbol names indexed", file=sys.stderr)

    print(f"[graph] resolving call / inherit edges by unique-name match ...", file=sys.stderr)
    stats = resolve_names(per_file, def_idx)
    print(f"[graph]   calls:    {stats['calls_resolved_unique']:>7d} / {stats['calls_total']:<7d} "
          f"({stats['calls_resolution_rate']*100:.1f}%)", file=sys.stderr)
    print(f"[graph]   inherits: {stats['inherits_resolved_unique']:>7d} / {stats['inherits_total']:<7d} "
          f"({stats['inherits_resolution_rate']*100:.1f}%)", file=sys.stderr)

    out_jsonl = args.out_dir / "graph.jsonl"
    with out_jsonl.open("w") as f:
        for fp in sorted(per_file):
            f.write(json.dumps(per_file[fp], ensure_ascii=False) + "\n")
    summary = {
        "clone_dir": str(args.clone_dir),
        "n_files_listed": len(py_files),
        "n_files_parsed": n_parsed,
        "n_files_failed": n_failed,
        "n_distinct_def_names": len(def_idx),
        "parse_seconds": dt,
        **stats,
    }
    (args.out_dir / "graph_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[graph] wrote {out_jsonl}", file=sys.stderr)
    print(f"[graph] wrote {args.out_dir/'graph_summary.json'}", file=sys.stderr)

    print("\n=== SUMMARY ===", file=sys.stderr)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<30s} = {v:.4f}", file=sys.stderr)
        else:
            print(f"  {k:<30s} = {v}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
