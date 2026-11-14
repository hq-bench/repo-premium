#!/usr/bin/env python3
"""What does the anchor ablation actually replace? (review #14 audit)

The paper describes the ablated tokens as "identifier tokens". The ablation
selects the K rarest QUERY tokens that also occur anywhere in the first gold
file -- comments and docstrings included -- so many of them may be prose.
This audit classifies every replaced token, for every ablated query in the
Python cells A/B and the R9 Rust/TypeScript slices:

  identifier_shaped  surface form has an underscore, a digit, an interior
                     capital, or is ALL-CAPS (code-looking on its face)
  dict_word          plain word found in /usr/share/dict/words
  other              neither (abbreviations, API names such as `qop`)

and, independently, whether the token occurs in the gold file's CODE
(outside comments and string literals) or only in its comments/strings:

  in_gold_code       occurs as a code token in the first gold file
  gold_prose_only    occurs in the gold file only inside comments/strings

It also reports replaced OCCURRENCES per query for the nonsense (ablated) and
placebo (control) arms -- the arms replace equal numbers of token TYPES but
not necessarily equal occurrences -- and the IDF gap between the two arms.

Runs where the clones live (GPU host); reads the existing ablation outputs only.
"""

from __future__ import annotations
from paths import DATA_ROOT

import argparse
import io
import json
import keyword
import re
import sys
import tokenize as pytok
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_retrieval as std  # noqa: E402

SLICES = [
    # name, audit dir (relative to data root), camelCase tokenization
    ("cellA", "ablation/qrels_cellA_ablated", False),
    ("cellB", "ablation/qrels_cellB_ablated", False),
    ("rust", "ablation_r9/rust", True),
    ("ts", "ablation_r9/ts", True),
]

_C_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_C_STRING = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`', re.S)


def identifier_shaped(surface: str) -> bool:
    return ("_" in surface or any(c.isdigit() for c in surface)
            or any(c.isupper() for c in surface[1:])
            or (surface.isupper() and len(surface) > 1))


def code_text_python(src: str) -> str:
    """Python source with comments and string literals removed."""
    out = []
    try:
        for tok in pytok.generate_tokens(io.StringIO(src).readline):
            if tok.type == pytok.NAME and not keyword.iskeyword(tok.string):
                out.append(tok.string)
    except (pytok.TokenError, IndentationError, SyntaxError):
        # Fall back to regex stripping on files the tokenizer rejects.
        src = re.sub(r'(?s)""".*?"""|\'\'\'.*?\'\'\'', " ", src)
        src = re.sub(r"#[^\n]*", " ", src)
        src = re.sub(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', " ", src)
        return src
    return " ".join(out)


def code_text_c_like(src: str) -> str:
    return _C_STRING.sub(" ", _C_COMMENT.sub(" ", src))


def surface_of(text: str, token: str, camel: bool) -> str:
    """First surface identifier in the query that tokenizes to `token`."""
    for m in std.TOKEN_RE.finditer(text):
        s = m.group(0)
        if (not camel and s.lower() == token) or (camel and token in std.tokenize(s)):
            return s
    return token


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA_ROOT)
    ap.add_argument("--dict", type=Path, default=DATA_ROOT / "inputs/words")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    words = {w.strip().lower() for w in args.dict.read_text(errors="ignore").splitlines()
             if w.strip().isalpha()}
    report: dict = {"dict": str(args.dict), "dict_size": len(words), "slices": {}}

    for name, rel, camel in SLICES:
        d = args.data / rel
        if not d.is_dir():
            report["slices"][name] = {"missing": str(d)}
            continue
        std.TOKENIZE_CAMELCASE = camel
        cats = Counter()
        loc = Counter()
        ctrl_cats = Counter()
        n_q = n_tok = 0
        occ_abl = occ_ctl = 0
        idf_abl, idf_ctl, idf_gap = [], [], []
        per_repo = {}
        for audit_path in sorted(d.glob("*_audit.jsonl")):
            repo = audit_path.name[: -len("_audit.jsonl")]
            qrows = {r["query_id"]: r for r in map(json.loads, (d / f"{repo}.jsonl")
                                                   .read_text().splitlines()) if r}
            clone = args.data / "repos" / repo
            code_cache: dict[str, set[str] | None] = {}
            r_tok = r_shape = 0
            for a in map(json.loads, audit_path.read_text().splitlines()):
                q = qrows.get(a["query_id"])
                if q is None:
                    continue
                text = q["problem_statement_stripped"]
                unabl = set(a.get("unablatable_tokens") or [])
                toks = [t for t, _ in a["ablated_tokens"] if t not in unabl]
                ctl = [t for t, _ in a["control_tokens"]]
                gold = q["gold_files"][0]
                if gold not in code_cache:
                    p = clone / gold
                    if p.is_file():
                        src = p.read_text(errors="ignore")
                        code = (code_text_python(src) if gold.endswith(".py")
                                else code_text_c_like(src))
                        code_cache[gold] = set(std.tokenize(code))
                    else:
                        code_cache[gold] = None
                gold_code = code_cache[gold]

                n_q += 1
                qt = std.tokenize(text)
                occ_abl += sum(t in set(toks) for t in qt)
                occ_ctl += sum(t in set(ctl) for t in qt)
                for (_, x), (_, y) in zip(a["ablated_tokens"], a["control_tokens"]):
                    idf_gap.append(abs(x - y))
                idf_abl += [i for _, i in a["ablated_tokens"]]
                idf_ctl += [i for _, i in a["control_tokens"]]

                for t in toks:
                    s = surface_of(text, t, camel)
                    n_tok += 1
                    r_tok += 1
                    if identifier_shaped(s):
                        cats["identifier_shaped"] += 1
                        r_shape += 1
                    elif t in words:
                        cats["dict_word"] += 1
                    else:
                        cats["other"] += 1
                    if gold_code is None:
                        loc["gold_unreadable"] += 1
                    elif t in gold_code:
                        loc["in_gold_code"] += 1
                    else:
                        loc["gold_prose_only"] += 1
                for t in ctl:
                    s = surface_of(text, t, camel)
                    ctrl_cats["identifier_shaped" if identifier_shaped(s)
                              else "dict_word" if t in words else "other"] += 1
            per_repo[repo] = {"tokens": r_tok,
                              "identifier_shaped_share": round(r_shape / r_tok, 4) if r_tok else None}

        share = lambda c, tot: {k: round(v / tot, 4) for k, v in sorted(c.items())} if tot else {}  # noqa: E731
        avg = lambda v: round(sum(v) / len(v), 4) if v else None  # noqa: E731
        report["slices"][name] = {
            "queries": n_q,
            "replaced_token_types": n_tok,
            "mean_types_per_query": round(n_tok / n_q, 3) if n_q else None,
            "category_counts": dict(cats),
            "category_share": share(cats, n_tok),
            "gold_location_counts": dict(loc),
            "gold_location_share": share(loc, n_tok),
            "control_category_share": share(ctrl_cats, sum(ctrl_cats.values())),
            "mean_occurrences_per_query": {"ablated": round(occ_abl / n_q, 3) if n_q else None,
                                           "control": round(occ_ctl / n_q, 3) if n_q else None},
            "mean_idf": {"ablated": avg(idf_abl), "control": avg(idf_ctl)},
            "mean_abs_idf_gap_pairwise": avg(idf_gap),
            "per_repo": per_repo,
        }
        s = report["slices"][name]
        print(f"[{name}] q={n_q} tokens={n_tok} shares={s['category_share']} "
              f"loc={s['gold_location_share']} occ={s['mean_occurrences_per_query']} "
              f"idf_gap={s['mean_abs_idf_gap_pairwise']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
