"""R6 Step 3 — pure code-graph baseline for NL → file retrieval.

Tests the reviewer's claim that "code graph is the SOTA for code search":
extracts identifiers from the PR/issue text and looks them up in a graph
def_index (name → defining files). Scores each file by IDF-weighted hit count.

This is a *pure-graph* baseline — no embeddings, no BM25 — meant to sit next
to BM25 and dense in the main results table. The expectation: it will win on
queries that name-drop symbols (path/function search) and lose on descriptive
queries (the embedding-premium zone).

Output (--out-file): per-query JSONL
    {query_id, gold_files, graph_top10, graph_ndcg10, graph_recall10,
     n_identifiers, n_repo_hits}

Usage:
  python scripts/graph_search.py \
      --qrel-file ${REPO_PREMIUM_DATA:-data}/qrels/django__django.jsonl \
      --graph-jsonl ${REPO_PREMIUM_DATA:-data}/graph/django__django/graph.jsonl \
      --out-file ${REPO_PREMIUM_DATA:-data}/coreb_repo_graph_eval/cellA/django__django.jsonl \
      --n 1000
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# CamelCase, snake_case (>=2 chars with underscore), dotted.path identifiers,
# and plain lowercase tokens of len >= 4 that look code-y.
RE_CAMEL = re.compile(r"\b[A-Z][a-zA-Z0-9]*[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b")
RE_SNAKE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+\b")
RE_DOTTED = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]+){1,}\b")
RE_BACKTICK = re.compile(r"`([^`\n]+)`")
RE_GENERIC = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b")


def extract_identifiers(text: str) -> list[str]:
    """Pull out plausible code identifiers from issue/PR text.

    Strategy: union of CamelCase / snake_case / dotted.path / backticked /
    long-ish tokens. Then split dotted paths into components. Lowercased
    duplicates collapse (we lookup by raw name; the def index uses raw names).
    """
    out: set[str] = set()
    for rx in (RE_CAMEL, RE_SNAKE):
        for m in rx.findall(text):
            out.add(m)
    for m in RE_DOTTED.findall(text):
        for part in m.split("."):
            if len(part) >= 3:
                out.add(part)
    for m in RE_BACKTICK.findall(text):
        # backticks often wrap fragments like `User.is_authenticated`
        for sub in re.split(r"[^a-zA-Z0-9_]+", m):
            if len(sub) >= 3:
                out.add(sub)
    # Generic long tokens are noisy; keep only if also looks code-like
    # (has either a digit, underscore, or mixed case).
    for m in RE_GENERIC.findall(text):
        if any(c.isdigit() for c in m) or "_" in m \
                or (m != m.lower() and m != m.upper()):
            out.add(m)
    # Drop pure-uppercase shouts (constants are OK but words aren't).
    return sorted(out)


def load_def_index(graph_jsonl: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Returns (name → list of defining files, all_files)."""
    idx: dict[str, set[str]] = defaultdict(set)
    files: list[str] = []
    with graph_jsonl.open() as fh:
        for line in fh:
            rec = json.loads(line)
            fp = rec["file"]
            files.append(fp)
            for d in rec.get("defs", []):
                idx[d["name"]].add(fp)
    return {k: sorted(v) for k, v in idx.items()}, files


def load_queries(qrel_file: Path, n: int) -> list[dict]:
    rows: list[dict] = []
    with qrel_file.open() as fh:
        for line in fh:
            r = json.loads(line)
            if (r.get("all_files_exist")
                    and 1 <= r.get("n_gold_files", 0) <= 3
                    and not r.get("filter_reason")):
                rows.append(r)
            if len(rows) >= n:
                break
    return rows


def score_files(
    identifiers: list[str],
    def_idx: dict[str, list[str]],
    n_files: int,
) -> list[tuple[str, float]]:
    """IDF-weighted hit scoring. Returns sorted (file, score) desc."""
    file_score: Counter[str] = Counter()
    n_hits = 0
    for ident in identifiers:
        files = def_idx.get(ident)
        if not files:
            continue
        df = len(files)
        idf = math.log((n_files + 1) / (df + 1)) + 1.0
        for f in files:
            file_score[f] += idf
        n_hits += 1
    return file_score.most_common(), n_hits


def dcg(rels: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked: list[str], gold: set[str], k: int = 10) -> float:
    rels = [1.0 if f in gold else 0.0 for f in ranked[:k]]
    ideal = sorted(rels, reverse=True)
    # Ideal cap = min(k, |gold|).
    ideal = [1.0] * min(k, len(gold)) + [0.0] * max(0, k - len(gold))
    d = dcg(rels)
    idl = dcg(ideal)
    return d / idl if idl > 0 else 0.0


def recall_at_k(ranked: list[str], gold: set[str], k: int = 10) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qrel-file", type=Path, required=True)
    ap.add_argument("--graph-jsonl", type=Path, required=True)
    ap.add_argument("--out-file", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()

    args.out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[graph-search] loading def index from {args.graph_jsonl}",
          file=sys.stderr)
    def_idx, all_files = load_def_index(args.graph_jsonl)
    n_files = len(all_files)
    print(f"[graph-search]   {len(def_idx)} distinct symbol names over "
          f"{n_files} files", file=sys.stderr)

    queries = load_queries(args.qrel_file, args.n)
    print(f"[graph-search] {len(queries)} queries after filter", file=sys.stderr)

    out_lines: list[str] = []
    ndcg_sum = 0.0
    recall_sum = 0.0
    n_with_hit = 0
    n_zero = 0
    for q in queries:
        gold = set(q["gold_files"])
        text = q.get("problem_statement", "")
        idents = extract_identifiers(text)
        ranked_scored, n_hits = score_files(idents, def_idx, n_files)
        ranked = [f for f, _ in ranked_scored]
        if n_hits > 0:
            n_with_hit += 1
        if not ranked:
            n_zero += 1
        nd = ndcg_at_k(ranked, gold, k=10)
        rc = recall_at_k(ranked, gold, k=10)
        ndcg_sum += nd
        recall_sum += rc
        out_lines.append(json.dumps({
            "query_id": q["query_id"],
            "gold_files": q["gold_files"],
            "n_identifiers": len(idents),
            "n_repo_hits": n_hits,
            "graph_top10": ranked[:10],
            "graph_ndcg10": nd,
            "graph_recall10": rc,
        }))
    with args.out_file.open("w") as f:
        f.write("\n".join(out_lines) + "\n")

    n = max(len(queries), 1)
    print(f"[graph-search] wrote {args.out_file}", file=sys.stderr)
    print(f"  queries           = {len(queries)}", file=sys.stderr)
    print(f"  with >=1 repo-hit = {n_with_hit} ({n_with_hit/n*100:.1f}%)",
          file=sys.stderr)
    print(f"  empty rank        = {n_zero} ({n_zero/n*100:.1f}%)",
          file=sys.stderr)
    print(f"  macro nDCG@10     = {ndcg_sum/n:.4f}", file=sys.stderr)
    print(f"  macro recall@10   = {recall_sum/n:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
