"""Emit the per-slice premium-decay-law appendix table (LaTeX).

For each of the 33 slices: n, n_{tau>0}, pooled delta, within-repo
tau_top1 percentile slope with 95% bootstrap CI. Groups rows by slice
family. Paste output into the appendix of coreb_repo_paper.tex.

Usage:
  conda run -n llm python scripts/law_slice_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from law_deepdive import (  # noqa: E402
    CHUNK_DUPES, load_all, slice_family, xy_within_repo,
)
from law_mining import slope_ci  # noqa: E402

ENC = {"jina05": "jina-code-0.5b", "jina15": "jina-code-1.5b",
       "qwen06": "Qwen3-Emb-0.6B", "qwen4": "Qwen3-Emb-4B",
       "qwen8": "Qwen3-Emb-8B", "c2llm05": "C2LLM-0.5B",
       "c2llm7": "C2LLM-7B", "f2llm06": "F2LLM-0.6B",
       "f2llm4": "F2LLM-4B", "e5base": "E5-base", "e5large": "E5-large",
       "gteq15": "gte-Qwen2-1.5B", "gteq7": "gte-Qwen2-7B"}
POOL = {"py_cur": "Cur-Pre", "py_raw": "Raw-Post", "py_r23": "Raw-Pre'23",
        "py_r22": "Raw-Pre'22", "py_desc": "Desc", "py_ood": "OOD held-out",
        "js_vuejs": "JS (vuejs/core)", "ts": "TS (TypeScript)",
        "go": "Go (hugo)", "rust": "Rust (tokio)"}


def pretty(name: str) -> str:
    if name in CHUNK_DUPES:
        c = name.rsplit("_c", 1)[1]
        return f"Cur-Pre / jina-code-0.5b ($c{{=}}{c}$)"
    for pk, pv in sorted(POOL.items(), key=lambda kv: -len(kv[0])):
        if name.startswith(pk + "_") or name == pk:
            rest = name[len(pk) + 1:] if len(name) > len(pk) else ""
            enc = ENC.get(rest, rest or "jina-code-0.5b")
            return f"{pv} / {enc}"
    return name.replace("_", r"\_")


def main() -> None:
    slices = load_all()
    order = sorted(slices, key=lambda n: (slice_family(n), pretty(n)))
    fam_seen = None
    lines = []
    for name in order:
        rows = slices[name]
        d = np.array([r["delta"] for r in rows])
        x, y = xy_within_repo(rows, "tau_top1")
        fam = slice_family(name)
        if fam != fam_seen:
            lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{fam}}}}} \\")
            fam_seen = fam
        core = "" if name not in CHUNK_DUPES else r"$^\dagger$"
        if len(x) >= 20:
            b, lo, hi = slope_ci(x, y)
            slope = f"${b:+.3f}$ & $[{lo:+.3f},{hi:+.3f}]$"
        else:
            slope = r"-- & --"
        lines.append(
            rf"{pretty(name)}{core} & {len(rows)} & ${d.mean():+.3f}$ & "
            + slope + r" \\")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
