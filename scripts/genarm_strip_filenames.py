#!/usr/bin/env python3
"""File-name ablation of the generative-lexical arm (review finding #15).

The generator names the true gold file in 43--83% of expansions, so a
reviewer can ask whether generate-then-grep closes the dense premium only
because the expansion spells out the answer's file name. This script removes
every file reference from the generated anchors and re-scores BM25.

Removed from each expansion (the original query text is left untouched):
  * path-like tokens (containing / or \\);
  * tokens ending in a source/text file extension (.py .js .ts .rs ...);
  * bare module/file stems that exactly match a file basename (sans
    extension) in that repository's corpus;
  * dotted module names (a.b.c) whose leading components resolve to a corpus
    module path.

Subcommands
  strip    read the genarm expanded qrels, write qrels with a new
           `expanded_query_nofn` field plus per-query strip statistics.
           Verifies first that rebuilding `expanded_query` from the unstripped
           generation reproduces the stored field byte for byte.
  analyze  join BM25 results (control, cached expanded, rerun expanded,
           stripped) with the cached dense scores and report premiums,
           repository-clustered paired bootstrap CIs (the genarm procedure),
           and TOST at +/-delta for "stripping file names does not change the
           premium", Holm-corrected within the four-cell family.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

FIELDS = (
    "identifiers", "file_or_module_hints", "apis",
    "error_terms", "search_phrases", "hypothetical_code",
)
EXT_RE = re.compile(
    r"\.(py|pyi|pyx|js|jsx|mjs|ts|tsx|rs|go|java|rb|c|cc|cpp|cxx|h|hh|hpp|cs|php|"
    r"scala|kt|swift|m|mm|sh|txt|rst|cfg|toml|yaml|yml|json|ini|md|html|css|in)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
SPAN_RE = re.compile(r"[A-Za-z0-9_./\\-]+")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
STRIP_CHARS = "`'\"(),;:[]{}<>"
CELLS = ("A", "B", "RA", "RB")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def anchors_text(generation: dict[str, Any]) -> str:
    """Identical to expand_queries_generative.anchors_text."""
    parts: list[str] = []
    for field in FIELDS:
        value = generation.get(field)
        if field == "hypothetical_code":
            if value:
                parts.append(str(value))
        elif value:
            parts.append(" ".join(str(v) for v in value))
    return "\n".join(parts).strip()


class Corpus:
    def __init__(self, clone_dir: Path) -> None:
        self.stems: set[str] = set()
        self.module_suffixes: set[str] = set()
        for path in clone_dir.rglob("*.py"):
            rel = path.relative_to(clone_dir)
            if rel.parts and rel.parts[0] == ".git":
                continue
            stem = rel.stem
            if stem != "__init__":
                self.stems.add(stem)
                mod = list(rel.with_suffix("").parts)
            else:
                mod = list(rel.parent.parts)
            for i in range(len(mod)):
                self.module_suffixes.add("/".join(mod[i:]))

    def is_fileref(self, raw: str) -> bool:
        t = raw.strip(STRIP_CHARS).rstrip(".")
        if not t:
            return False
        if "/" in t or "\\" in t:
            return True
        if EXT_RE.search(t):
            return True
        if t in self.stems:
            return True
        if "." in t:
            # Leading identifier components only, so `libgdal.so.3.13` is
            # judged on `libgdal`.
            parts = []
            for p in t.split("."):
                if not IDENT_RE.match(p):
                    break
                parts.append(p)
            if parts:
                # k=1 covers `autoreload.main()`: the leading component alone
                # names a module.
                for k in range(len(parts), 0, -1):
                    if "/".join(parts[:k]) in self.module_suffixes:
                        return True
        return False


def gold_named(text: str, gold: list[str]) -> bool:
    """Does `text` name a gold file by path, basename, stem or dotted module?"""
    golds = [g.strip("./") for g in gold]
    stems = {Path(g).stem for g in golds if Path(g).stem != "__init__"}
    bases = {Path(g).name for g in golds}
    mods = set()
    for g in golds:
        p = Path(g)
        parts = list(p.with_suffix("").parts) if p.stem != "__init__" else list(p.parent.parts)
        for i in range(len(parts)):
            mods.add("/".join(parts[i:]))
    for raw in SPAN_RE.findall(text):
        t = raw.strip(STRIP_CHARS).rstrip(".").replace("\\", "/").lstrip("./")
        if not t:
            continue
        if t in stems or t in bases:
            return True
        if any(g == t or g.endswith("/" + t) for g in golds):
            return True
        cand = t[:-3] if t.endswith(".py") else t
        if "." in cand and "/" not in cand:
            cand = cand.replace(".", "/")
        segs = cand.split("/")
        for k in range(len(segs), 0, -1):
            if "/".join(segs[:k]) in mods and k >= 1 and "/".join(segs[:k]) not in ("",):
                if k == 1 and segs[0] not in stems:
                    continue
                return True
    return False


def hint_gold_predicted(gen: dict[str, Any], gold: list[str]) -> bool:
    """The paper's measure (analyze_genarm.grounding)."""
    hints = [str(h).strip().lstrip("./") for h in gen.get("file_or_module_hints", [])]
    return any(g == h or g.endswith("/" + h) for g in gold for h in hints if h)


def strip_generation(gen: dict[str, Any], corpus: Corpus) -> tuple[dict[str, Any], int, int]:
    out: dict[str, Any] = {}
    removed = total = 0

    def sub_span(m: re.Match) -> str:
        nonlocal removed, total
        total += 1
        if corpus.is_fileref(m.group(0)):
            removed += 1
            return ""
        return m.group(0)

    for field in FIELDS:
        value = gen.get(field)
        if field == "hypothetical_code":
            out[field] = SPAN_RE.sub(sub_span, str(value or ""))
            continue
        # Sub-token spans, not whitespace tokens, so a stem quoted inside a
        # token (`format="html"`) or an extension followed by an anchor
        # (`compiler.py#L428`) is still caught.
        kept_entries = []
        for entry in value or []:
            new = SPAN_RE.sub(sub_span, str(entry))
            new = " ".join(new.split())
            if new.strip(STRIP_CHARS + "=#.-"):
                kept_entries.append(new)
        out[field] = kept_entries
    return out, removed, total


def cmd_strip(args: argparse.Namespace) -> int:
    jobs = [line.split("\t") for line in Path(args.jobs).read_text().splitlines() if line.strip()]
    targets = {}
    for arm, cell, repo, qrel, clone in jobs:
        if arm == "expanded":
            targets[(cell, repo)] = (Path(qrel), Path(clone))
    stats: dict[str, Any] = {}
    out_root = Path(args.out_root)
    for (cell, repo), (qrel, clone) in sorted(targets.items()):
        corpus = Corpus(clone)
        rows = read_jsonl(qrel)
        out_rows = []
        s = stats.setdefault(cell, defaultdict(int))
        for row in rows:
            gen = row.get("generation")
            original = row["problem_statement_stripped"]
            if gen is None:
                assert row["expanded_query"] == original
                new = dict(row, expanded_query_nofn=original)
                s["n_failed_generation"] += 1
                out_rows.append(new)
                continue
            rebuilt = (original + "\n" + anchors_text(gen)).strip()
            if rebuilt != row["expanded_query"]:
                raise SystemExit(f"rebuild mismatch {row['query_id']}")
            stripped, removed, total = strip_generation(gen, corpus)
            anchors_new = anchors_text(stripped)
            gold = row.get("gold_files") or []
            before_text = gold_named(anchors_text(gen), gold)
            after_text = gold_named(anchors_new, gold)
            s["n_queries"] += 1
            s["tokens_total"] += total
            s["tokens_removed"] += removed
            s["expansions_affected"] += int(removed > 0)
            s["gold_named_hint_before"] += int(hint_gold_predicted(gen, gold))
            s["gold_named_hint_after"] += int(hint_gold_predicted(stripped, gold))
            s["gold_named_text_before"] += int(before_text)
            s["gold_named_text_after"] += int(after_text)
            out_rows.append(dict(
                row,
                generation_nofn=stripped,
                expanded_query_nofn=(original + "\n" + anchors_new).strip(),
            ))
        dest = out_root / f"qrels_cell{cell}_nofn" / qrel.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
        print(f"cell{cell} {repo}: {len(out_rows)} rows", flush=True)
    stats = {c: dict(v) for c, v in stats.items()}
    Path(args.stats_out).write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


# ---------------------------------------------------------------- analysis

def load_arm(base: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for repo_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        path = repo_dir / "smoke_test_django.jsonl"
        if path.is_file():
            for rec in read_jsonl(path):
                rec["repo"] = repo_dir.name
                out[rec["query_id"]] = rec
    return out


def cluster_draws(clusters: list[list[float]], n_boot: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(clusters)
    draws = []
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in range(n):
            sample.extend(clusters[rng.randrange(n)])
        draws.append(sum(sample) / len(sample))
    draws.sort()
    return draws


def q(draws: list[float], p: float) -> float:
    return draws[min(len(draws) - 1, int(p * len(draws)))]


def by_repo(rows: list[dict[str, Any]], key: str) -> list[list[float]]:
    g: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        g[r["repo"]].append(r[key])
    return list(g.values())


def cmd_analyze(args: argparse.Namespace) -> int:
    results = Path(args.results_root)
    report: dict[str, Any] = {"delta": args.delta, "n_boot": args.n_boot, "seed": args.seed,
                              "cells": {}}
    pvals = {}
    for cell, canon in args.cell:
        ctrl = load_arm(results / f"control_cell{cell}")
        exp_cached = load_arm(results / f"expanded_cell{cell}")
        exp_rerun = load_arm(Path(args.new_root) / f"expanded_cell{cell}")
        nofn = load_arm(Path(args.new_root) / f"nofn_cell{cell}")
        dense = {}
        for repo_dir in sorted(p for p in Path(canon).iterdir() if p.is_dir()):
            f = repo_dir / "smoke_test_django.jsonl"
            if f.is_file():
                for rec in read_jsonl(f):
                    dense[rec["query_id"]] = rec["dense_ndcg10"]
        common = sorted(set(ctrl) & set(exp_cached) & set(dense) & set(nofn) & set(exp_rerun))
        guard_max = max(abs(exp_rerun[k]["bm25_ndcg10"] - exp_cached[k]["bm25_ndcg10"])
                        for k in common)
        rows = []
        for k in common:
            d = dense[k]
            rows.append({
                "repo": ctrl[k]["repo"],
                "premium_control": d - ctrl[k]["bm25_ndcg10"],
                "premium_expanded": d - exp_cached[k]["bm25_ndcg10"],
                "premium_nofn": d - nofn[k]["bm25_ndcg10"],
                "change": exp_cached[k]["bm25_ndcg10"] - nofn[k]["bm25_ndcg10"],
                "bm25_expanded": exp_cached[k]["bm25_ndcg10"],
                "bm25_nofn": nofn[k]["bm25_ndcg10"],
            })
        st: dict[str, Any] = {"n_queries": len(rows), "n_repos": len({r["repo"] for r in rows}),
                              "guard_max_abs_diff_expanded_rerun_vs_cached": guard_max}
        for key in ("premium_control", "premium_expanded", "premium_nofn"):
            dr = cluster_draws(by_repo(rows, key), args.n_boot, args.seed)
            pt = sum(r[key] for r in rows) / len(rows)
            st[key] = {"mean": round(pt, 4), "ci95": [round(q(dr, .025), 4), round(q(dr, .975), 4)]}
        for key in ("bm25_expanded", "bm25_nofn"):
            st[f"mean_{key}"] = round(sum(r[key] for r in rows) / len(rows), 4)
        # change = premium_nofn - premium_expanded = bm25_expanded - bm25_nofn
        dr = cluster_draws(by_repo(rows, "change"), args.n_boot, args.seed)
        pt = sum(r["change"] for r in rows) / len(rows)
        lo90, hi90 = q(dr, .05), q(dr, .95)
        # TOST bootstrap p: max of the two one-sided tail fractions
        p_lower = sum(1 for x in dr if x <= -args.delta) / len(dr)
        p_upper = sum(1 for x in dr if x >= args.delta) / len(dr)
        p_tost = max(p_lower, p_upper)
        prem_c = st["premium_control"]["mean"]
        st["change_in_premium"] = {
            "mean": round(pt, 4), "ci90": [round(lo90, 4), round(hi90, 4)],
            "ci95": [round(q(dr, .025), 4), round(q(dr, .975), 4)],
            "tost_p": p_tost, "equivalent_unadjusted": lo90 > -args.delta and hi90 < args.delta,
        }
        st["closed_fraction_expanded"] = round((prem_c - st["premium_expanded"]["mean"]) / prem_c, 4)
        st["closed_fraction_nofn"] = round((prem_c - st["premium_nofn"]["mean"]) / prem_c, 4)
        pvals[cell] = p_tost
        report["cells"][cell] = st
    # Holm within the family
    order = sorted(pvals, key=lambda c: pvals[c])
    m = len(order)
    running = 0.0
    for i, c in enumerate(order):
        adj = min(1.0, (m - i) * pvals[c])
        running = max(running, adj)
        report["cells"][c]["change_in_premium"]["tost_p_holm"] = round(running, 4)
        report["cells"][c]["change_in_premium"]["equivalent_holm"] = running < 0.05
    if args.strip_stats:
        report["strip_stats"] = json.loads(Path(args.strip_stats).read_text())
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("strip")
    s.add_argument("--jobs", required=True)
    s.add_argument("--out-root", required=True)
    s.add_argument("--stats-out", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--results-root", required=True, help="original genarm results")
    a.add_argument("--new-root", required=True, help="rerun expanded + nofn results")
    a.add_argument("--cell", action="append", nargs=2, required=True, metavar=("NAME", "CANON"))
    a.add_argument("--strip-stats")
    a.add_argument("--delta", type=float, default=0.05)
    a.add_argument("--n-boot", type=int, default=10000)
    a.add_argument("--seed", type=int, default=20260813)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    return cmd_strip(args) if args.cmd == "strip" else cmd_analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
