#!/usr/bin/env python3
"""Build a source-only mirror of ${REPO_PREMIUM_DATA:-data} for re-scoring every analysis.

Gold sets in the raw pools (merged-PR file lists) keep the test files the PR edited, while
SWE-bench-derived gold sets do not. The paper's task is source localization, so the primary
labels drop test files everywhere. This script copies ${REPO_PREMIUM_DATA:-data} to a mirror in which

  * every JSON/JSONL record with a ``gold_files`` list has its test files removed
    (``n_gold_files`` updated); records left with no gold file are dropped;
  * every stored ``<ret>_ndcg10`` / ``<ret>_recall10`` with a matching ``<ret>_top10`` is
    recomputed from that top-10 (exact for @10 metrics), and ``delta_ndcg10`` with it.

Retrieved test files then count as non-relevant; the corpus is unchanged. Analysis scripts
run with REPO_PREMIUM_DATA set to the mirror.

With ``--indexed``, gold files the corpus does not index are removed as well (docs,
changelogs, templates and other non-code files a PR edited). Python pools index ``.py`` files;
the multi-language pools (R9 slices, Multi-SWE-bench, DeepSWE, scale-up, lockbox) index the
union of their code extensions. This is the paper's primary label: the indexed, non-test
files the resolving change edited.

Usage: python scripts/build_source_only_mirror.py [--out DIR] [--indexed]
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
import math
import re
import shutil
from pathlib import Path

SRC = DATA_ROOT
SKIP_DIRS = {"local_inputs_202610"}  # cluster inputs, not read by analyses

TEST_DIR = re.compile(r"(^|/)(tests?|testing|__tests__|testdata|testscripts|unittests|__snapshots__)/")
TEST_FILE = re.compile(
    r"(^|/)(test_[^/]*\.py|[^/]*_test\.py|conftest\.py|[^/]*_test\.go"
    r"|[^/]*\.(test|spec)\.[cm]?[jt]sx?|[^/]*\.snap|[^/]*Tests?\.java)$"
)


def is_test(path: str) -> bool:
    return bool(TEST_DIR.search(path) or TEST_FILE.search(path))


PY_EXTS = (".py",)
MULTI_EXTS = (".py", ".pyi", ".go", ".rs", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".java")
# data paths whose corpora are not Python-only
MULTI_PATH = re.compile(r"r9|_ts|_go|_rs|rust|tokio|hugo|vuejs|TypeScript|darkreader|mui|deepswe|scaleup|"
                        r"ml_sweep|cur_|lockbox|java", re.I)
INDEXED = False   # set by --indexed
CTX_EXTS = PY_EXTS  # set per file


def ctx_exts(rel: str) -> tuple[str, ...]:
    """Extensions the corpus behind this data path indexes (the R9 runners' --code-exts)."""
    if re.search(r"r9c_go|r9_go|hugo|_go_|_go\.|split_r9_go|plain_r9_go|guard_r9_go", rel):
        return (".go",)
    if re.search(r"r9d_rust|r9_rust|tokio|_rs_|_rs\.|r9_rs", rel):
        return (".rs",)
    if re.search(r"ml_sweep|cur_.*_ts|raw_vuejs_ts|darkreader|mui", rel):
        return (".ts", ".js", ".vue", ".tsx", ".jsx")
    if re.search(r"r9b_ts|r9_ts|r9_vuejs|vuejs|TypeScript", rel):
        return (".ts", ".tsx", ".js", ".mjs", ".cjs")
    return MULTI_EXTS if MULTI_PATH.search(rel) else PY_EXTS


def keep(path: str) -> bool:
    if is_test(path):
        return False
    return path.endswith(CTX_EXTS) if INDEXED else True


def ndcg10(top: list[str], gold: set[str]) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, f in enumerate(top[:10]) if f in gold)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(len(gold), 10)))


def fix(rec: dict, stats: dict) -> dict | None:
    gold = rec["gold_files"]
    if not isinstance(gold, list) or not all(isinstance(g, str) for g in gold):
        return rec
    src = [g for g in gold if keep(g)]
    stats["records"] += 1
    stats["test_files"] += len(gold) - len(src)
    stats["changed"] += len(src) != len(gold)
    if not src:
        stats["dropped"] += 1
        return None
    if len(src) == len(gold):
        return rec
    rec = dict(rec, gold_files=src)
    if "n_gold_files" in rec:
        rec["n_gold_files"] = len(src)
    g = set(src)
    for k in list(rec):
        if k.endswith("_top10") and isinstance(rec[k], list):
            ret = k[: -len("_top10")]
            if f"{ret}_ndcg10" in rec:
                rec[f"{ret}_ndcg10"] = ndcg10(rec[k], g)
            if f"{ret}_recall10" in rec:
                rec[f"{ret}_recall10"] = len(g & set(rec[k][:10])) / len(g)
    if "delta_ndcg10" in rec and "dense_ndcg10" in rec and "bm25_ndcg10" in rec:
        rec["delta_ndcg10"] = rec["dense_ndcg10"] - rec["bm25_ndcg10"]
    return rec


def walk(obj, stats):
    """Fix gold-bearing dicts anywhere inside a JSON document; None marks a dropped record."""
    if isinstance(obj, dict):
        if "gold_files" in obj:
            obj = fix(obj, stats)
            if obj is None:
                return None
        return {k: (w if (w := walk(v, stats)) is not None or v is None else v) for k, v in obj.items()}
    if isinstance(obj, list):
        out = [walk(v, stats) for v in obj]
        return [v for v, o in zip(out, obj) if v is not None or o is None]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_ROOT.with_name(DATA_ROOT.name + "_source_only")))
    ap.add_argument("--indexed", action="store_true")
    args = ap.parse_args()
    global INDEXED, CTX_EXTS
    INDEXED = args.indexed
    out = Path(args.out).resolve()
    source = SRC.resolve()
    if out == source or out in source.parents or source in out.parents:
        ap.error("--out and the source data directory must not overlap")
    if out.exists():
        shutil.rmtree(out)
    stats = {"records": 0, "changed": 0, "test_files": 0, "dropped": 0, "files_rewritten": 0}
    for p in SRC.rglob("*"):
        rel = p.relative_to(SRC)
        if rel.parts and rel.parts[0] in SKIP_DIRS:
            continue
        dst = out / rel
        CTX_EXTS = ctx_exts(str(rel))
        if p.is_symlink():  # copy links verbatim (two point at cluster paths)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(p.readlink())
            continue
        if p.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = None
        if p.suffix in {".jsonl", ".json"}:
            try:
                text = p.read_text()
            except UnicodeDecodeError:
                text = None
        if text is None or "gold_files" not in text:
            shutil.copy2(p, dst)
            continue
        if p.suffix == ".jsonl":
            lines = []
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    lines.append(line)
                    continue
                r = walk(r, stats)
                if r is not None:
                    lines.append(json.dumps(r))
            dst.write_text("\n".join(lines) + ("\n" if lines else ""))
        else:
            try:
                doc = json.loads(text)
            except json.JSONDecodeError:
                shutil.copy2(p, dst)
                continue
            dst.write_text(json.dumps(walk(doc, stats), indent=1))
        stats["files_rewritten"] += 1
    print(json.dumps(stats))
    (out / "SOURCE_ONLY_MIRROR.json").write_text(json.dumps(
        {**stats, "test_dir": TEST_DIR.pattern, "test_file": TEST_FILE.pattern, "indexed": INDEXED,
         "py_exts": PY_EXTS, "multi_exts": MULTI_EXTS, "multi_path": MULTI_PATH.pattern}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
