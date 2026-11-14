#!/usr/bin/env python3
"""Analyse the generative-lexical arm.

Tests the reviewer's conjecture that lexical retrieval on a natural-language
query is really an LLM performing the text-to-code mapping, leaving BM25 a
code-to-code hard match. Three questions:

Q-A  Does generating lexical anchors before BM25 close the dense premium?
     Compares BM25 on the original query, BM25 on original+anchors, and BM25 on
     anchors alone, against the cached dense score for the same query.

Q-B  Does it work by moving the query along the decay curve? Each arm's own
     tau_q is recomputed by the retrieval script from that arm's query text, so
     tau movement is read directly rather than modelled. If generation
     substitutes for the encoder, expanded queries should sit at higher tau and
     their premium should shrink accordingly.

Q-C  How lossy is generation? Reports the fraction of generated path hints and
     identifiers that actually occur in the repository, since an anchor that
     does not exist cannot help BM25 and is the mechanism by which the
     substitution would fail.

Confidence intervals are repository-clustered paired bootstrap.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
ARMS = ("control", "expanded", "generated")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def cluster_bootstrap(
    clusters: list[list[float]], n_boot: int, seed: int
) -> tuple[float, float, float]:
    flat = [v for c in clusters for v in c]
    if not flat:
        return float("nan"), float("nan"), float("nan")
    point = mean(flat)
    rng = random.Random(seed)
    draws = []
    n = len(clusters)
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in range(n):
            sample.extend(clusters[rng.randrange(n)])
        if sample:
            draws.append(mean(sample))
    draws.sort()
    if not draws:
        return point, float("nan"), float("nan")
    return point, draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))]


def by_repo(rows: list[dict[str, Any]], key: str) -> list[list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            groups[row["repo"]].append(row[key])
    return list(groups.values())


def load_arm(root: Path, arm: str, cell: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    base = root / f"{arm}_cell{cell}"
    if not base.is_dir():
        return out
    for repo_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        path = repo_dir / "smoke_test_django.jsonl"
        if not path.is_file():
            continue
        for record in read_jsonl(path):
            record["repo"] = repo_dir.name
            out[record["query_id"]] = record
    return out


def load_cached_dense(canon_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not canon_root.is_dir():
        return out
    for repo_dir in sorted(p for p in canon_root.iterdir() if p.is_dir()):
        path = repo_dir / "smoke_test_django.jsonl"
        if not path.is_file():
            continue
        for record in read_jsonl(path):
            out[record["query_id"]] = record
    return out


def corpus_vocab(clone_dir: Path, exts: tuple[str, ...]) -> tuple[set[str], set[str]]:
    """Return (relative paths, identifier token vocabulary) for one repository."""
    paths: set[str] = set()
    vocab: set[str] = set()
    for ext in exts:
        for path in clone_dir.rglob(f"*{ext}"):
            rel = path.relative_to(clone_dir).as_posix()
            paths.add(rel)
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            vocab.update(t.lower() for t in TOKEN_RE.findall(text))
    return paths, vocab


def grounding(
    expanded_rows: list[dict[str, Any]], repos_root: Path, exts: tuple[str, ...]
) -> dict[str, Any]:
    by_repo_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in expanded_rows:
        by_repo_rows[row["repo"]].append(row)

    path_hits = path_total = ident_hits = ident_total = 0
    gold_predicted = gold_total = 0
    for repo, rows in sorted(by_repo_rows.items()):
        clone = repos_root / repo
        if not clone.is_dir():
            continue
        paths, vocab = corpus_vocab(clone, exts)
        for row in rows:
            gen = row.get("generation") or {}
            for hint in gen.get("file_or_module_hints", []):
                path_total += 1
                needle = str(hint).strip().lstrip("./")
                if needle and any(needle == p or p.endswith("/" + needle) for p in paths):
                    path_hits += 1
            for ident in gen.get("identifiers", []) + gen.get("apis", []):
                for token in TOKEN_RE.findall(str(ident)):
                    ident_total += 1
                    if token.lower() in vocab:
                        ident_hits += 1
            gold = row.get("gold_files") or []
            if gold:
                gold_total += 1
                hints = [str(h).strip().lstrip("./") for h in gen.get("file_or_module_hints", [])]
                if any(g == h or g.endswith("/" + h) for g in gold for h in hints if h):
                    gold_predicted += 1
    return {
        "path_hint_grounding": round(path_hits / path_total, 4) if path_total else None,
        "n_path_hints": path_total,
        "identifier_grounding": round(ident_hits / ident_total, 4) if ident_total else None,
        "n_identifier_tokens": ident_total,
        "gold_path_predicted_rate": round(gold_predicted / gold_total, 4) if gold_total else None,
        "n_queries_with_gold": gold_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    # Per-cell rather than global: the pre-cutoff negative-control cells read a
    # different corpus snapshot (repos_2024q1) than the main cells (repos).
    parser.add_argument("--cell", action="append", nargs=4, required=True,
                        metavar=("NAME", "CANON_DIR", "EXPANDED_QRELS", "REPOS_ROOT"))
    parser.add_argument("--code-exts", default=".py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    exts = tuple(e.strip() for e in args.code_exts.split(",") if e.strip())
    report: dict[str, Any] = {"n_boot": args.n_boot, "cells": {}}

    for cell, canon_dir, qrel_dir_s, repos_root_s in args.cell:
        canon_root, qrel_dir = Path(canon_dir), Path(qrel_dir_s)
        repos_root = Path(repos_root_s)
        arms = {a: load_arm(args.results_root, a, cell) for a in ARMS}
        cached = load_cached_dense(canon_root)
        common = set(arms["control"]) & set(arms["expanded"]) & set(arms["generated"]) & set(cached)
        if not common:
            report["cells"][cell] = {"error": "no overlapping queries", "counts":
                                     {a: len(arms[a]) for a in ARMS} | {"cached": len(cached)}}
            continue

        rows: list[dict[str, Any]] = []
        for qid in sorted(common):
            ctrl, exp, gen_, cache = (arms["control"][qid], arms["expanded"][qid],
                                      arms["generated"][qid], cached[qid])
            dense = cache.get("dense_ndcg10")
            rows.append({
                "query_id": qid,
                "repo": ctrl["repo"],
                "dense": dense,
                "bm25_control": ctrl.get("bm25_ndcg10"),
                "bm25_expanded": exp.get("bm25_ndcg10"),
                "bm25_generated": gen_.get("bm25_ndcg10"),
                "tau_control": ctrl.get("tau_q"),
                "tau_expanded": exp.get("tau_q"),
                "d_tau": (exp.get("tau_q") or 0.0) - (ctrl.get("tau_q") or 0.0),
                "gain_expanded": (exp.get("bm25_ndcg10") or 0.0) - (ctrl.get("bm25_ndcg10") or 0.0),
                "gain_generated": (gen_.get("bm25_ndcg10") or 0.0) - (ctrl.get("bm25_ndcg10") or 0.0),
                "premium_control": (dense or 0.0) - (ctrl.get("bm25_ndcg10") or 0.0),
                "premium_expanded": (dense or 0.0) - (exp.get("bm25_ndcg10") or 0.0),
            })

        stats: dict[str, Any] = {"n_queries": len(rows),
                                 "n_repositories": len({r["repo"] for r in rows})}
        for key in ("dense", "bm25_control", "bm25_expanded", "bm25_generated",
                    "tau_control", "tau_expanded"):
            stats[f"mean_{key}"] = round(mean([r[key] for r in rows if r[key] is not None]), 4)
        for key in ("gain_expanded", "gain_generated", "d_tau",
                    "premium_control", "premium_expanded"):
            point, lo, hi = cluster_bootstrap(by_repo(rows, key), args.n_boot, args.seed)
            stats[key] = {"mean": round(point, 4), "ci": [round(lo, 4), round(hi, 4)]}

        # Fraction of the original dense premium that expansion recovers.
        prem_c, prem_e = stats["premium_control"]["mean"], stats["premium_expanded"]["mean"]
        stats["premium_closed_fraction"] = (
            round((prem_c - prem_e) / prem_c, 4) if prem_c else None)

        # Does the tau shift explain the BM25 gain? Pearson r over queries.
        pairs = [(r["d_tau"], r["gain_expanded"]) for r in rows]
        n = len(pairs)
        mx, my = mean([p[0] for p in pairs]), mean([p[1] for p in pairs])
        cov = sum((x - mx) * (y - my) for x, y in pairs)
        vx = sum((x - mx) ** 2 for x, _ in pairs)
        vy = sum((y - my) ** 2 for _, y in pairs)
        stats["r_dtau_vs_gain"] = round(cov / (vx * vy) ** 0.5, 4) if vx and vy else None
        stats["n_pairs"] = n

        expanded_qrels = [r for f in sorted(qrel_dir.glob("*.jsonl"))
                          for r in read_jsonl(f)]
        for row in expanded_qrels:
            row["repo"] = row.get("repo", "")
        keep = {r["query_id"] for r in rows}
        repo_of = {r["query_id"]: r["repo"] for r in rows}
        sel = [dict(r, repo=repo_of[r["query_id"]])
               for r in expanded_qrels if r["query_id"] in keep]
        stats["generation_grounding"] = grounding(sel, repos_root, exts)
        report["cells"][cell] = stats

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
