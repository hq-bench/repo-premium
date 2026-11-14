#!/usr/bin/env python3
"""Which part of the LLM expansion closes the dense premium?

The generator emits six fields per query: identifiers, file_or_module_hints, apis,
error_terms, search_phrases (natural-language prose) and hypothetical_code. This script
builds query variants that append one group of fields to the stripped query and scores
whole-file BM25 on each:

  full     the released expanded query (guard: must reproduce gen_sweep202610 exactly)
  lexical  identifiers + apis + error_terms + file_or_module_hints
  nopath   identifiers + apis + error_terms (no file names)
  prose    search_phrases only
  code     hypothetical_code only
  words    search_phrases with every identifier-shaped token (snake/camel case, dotted,
           call syntax, ALLCAPS) and every token also listed in another field removed

`build` writes qrels with a `variant_query` field per variant; `analyze` re-scores each run
with the paper's labels (indexed, non-test files) against the canonical dense run.

Usage:
  python scripts/anchor_type_split.py build
  bash  <scratch>/run_anchor_split.sh       # BM25 runs (smoke_test_django.py --skip-dense)
  python scripts/anchor_type_split.py analyze
"""
from __future__ import annotations
from paths import DATA_ROOT

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_source_only_mirror import is_test  # noqa: E402

D = DATA_ROOT
INP = D / "local_inputs_202610"
OUT_Q = INP / "anchor_split_qrels"
RUNS = D / "smoke_test_results/anchor_split202610"
CELLS = ("A", "B", "RA", "RB")
DENSE = {"A": "canon202607/jina05_cellA_c1000", "B": "canon202607/jina05_cellB_c1000",
         "RA": "r8_robust_jina-0.5b_c1000", "RB": "r8_2022_jina-0.5b_c1000"}
GEN = "gen_sweep202610/gen_cell{c}_c1000000000"
BM25W = {"A": "bm25_sweep202610/bm25_cellA_c1000000000", "B": "bm25_sweep202610/bm25_cellB_c1000000000",
         "RA": "gen_sweep202610/control_cellRA_c1000000000", "RB": "gen_sweep202610/control_cellRB_c1000000000"}
VARIANTS = {
    "lexical": ("identifiers", "apis", "error_terms", "file_or_module_hints"),
    "nopath": ("identifiers", "apis", "error_terms"),
    "prose": ("search_phrases",),
    "code": ("hypothetical_code",),
}


import re
IDENT = re.compile(r"[_.()\[\]]|[a-z][A-Z]|^[A-Z0-9]{3,}$|\d")


def words_only(gen: dict) -> str:
    """Prose phrases with identifier-shaped tokens and tokens from the anchor fields removed."""
    listed = set()
    for f in ("identifiers", "apis", "error_terms", "file_or_module_hints"):
        for x in gen.get(f) or []:
            listed.update(t.lower() for t in re.findall(r"\w+", str(x)))
    out = []
    for ph in gen.get("search_phrases") or []:
        toks = [t for t in str(ph).split() if not IDENT.search(t) and t.strip(",.:;").lower() not in listed]
        if toks:
            out.append(" ".join(toks))
    return " ".join(out)


def text_of(gen: dict, fields: tuple[str, ...]) -> str:
    parts = []
    for f in fields:
        v = gen.get(f)
        if isinstance(v, list):
            parts.append(" ".join(str(x) for x in v))
        elif isinstance(v, str):
            parts.append(v)
    return "\n".join(p for p in parts if p.strip())


def build() -> None:
    for c in CELLS:
        src = INP / f"qrels_cell{c}_expanded"
        for f in sorted(src.glob("*.jsonl")):
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            for v, fields in [("full", None), ("words", None), *VARIANTS.items()]:
                out = OUT_Q / f"cell{c}_{v}" / f.name
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("w") as fh:
                    for r in rows:
                        g = r.get("generation") or {}
                        base = r.get("problem_statement_stripped") or r.get("problem_statement", "")
                        r = dict(r)
                        if v == "full":
                            r["variant_query"] = r["expanded_query"]
                        elif v == "words":
                            extra = words_only(g) if g else ""
                            r["variant_query"] = base + ("\n" + extra if extra else "")
                        else:
                            extra = text_of(g, fields) if g else ""
                            r["variant_query"] = base + ("\n" + extra if extra else "")
                        fh.write(json.dumps(r) + "\n")
    print("wrote", OUT_Q)


def load(d: Path) -> dict:
    out = {}
    for f in d.glob("*/smoke_test_django.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line); r["_repo"] = f.parent.name; out[r["query_id"]] = r
    return out


def ndcg(top, gold):
    g = set(gold)
    return sum(1 / math.log2(i + 2) for i, f in enumerate(top[:10]) if f in g) / sum(
        1 / math.log2(i + 2) for i in range(min(len(g), 10)))


def rci(rows):
    groups = {}
    for rp, v in rows:
        groups.setdefault(rp, []).append(v)
    ks = sorted(groups); rng = random.Random(20261011); d = []
    for _ in range(10000):
        s = [v for _ in ks for v in groups[ks[rng.randrange(len(ks))]]]
        d.append(sum(s) / len(s))
    d.sort()
    return [d[250], d[9750]]


def analyze() -> None:
    SM = D / "smoke_test_results"
    res = {"guard": {}, "pools": {}}
    for c in CELLS:
        dense = load(SM / DENSE[c]); gen = load(SM / GEN.format(c=c)); bm = load(SM / BM25W[c])
        runs = {v: load(RUNS / f"cell{c}_{v}") for v in ["full", *VARIANTS, "words"]}
        common = [k for k in runs["full"] if k in gen]
        res["guard"][c] = max(abs(runs["full"][k]["bm25_ndcg10"] - gen[k]["bm25_ndcg10"]) for k in common) if common else None
        pool = {}
        keys = [k for k in dense if k in bm and all(k in runs[v] for v in runs)]
        rows = {v: [] for v in ["bm25", *runs]}
        for k in keys:
            gold = [g for g in dense[k]["gold_files"] if not is_test(g) and g.endswith(".py")]
            if not gold:
                continue
            d = ndcg(dense[k]["dense_top10"], gold)
            rows["bm25"].append((dense[k]["_repo"], d - ndcg(bm[k]["bm25_top10"], gold)))
            for v in runs:
                rows[v].append((dense[k]["_repo"], d - ndcg(runs[v][k]["bm25_top10"], gold)))
        for v, rr in rows.items():
            pool[v] = {"n": len(rr), "premium": sum(x for _, x in rr) / len(rr), "ci95": rci(rr)}
        res["pools"][c] = pool
        print(c, "guard", res["guard"][c], " ".join(f"{v}:{p['premium']:+.3f}[{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}]" for v, p in pool.items()))
    (D / "analysis/indexed/anchor_type_split.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    {"build": build, "analyze": analyze}[sys.argv[1]]()
