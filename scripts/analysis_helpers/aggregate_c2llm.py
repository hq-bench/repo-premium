#!/usr/bin/env python3
"""Aggregate C2LLM smoke_test_django_summary.json across repos for Cells A/B x 0.5B/7B."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_ROOT
import json
import sys
from pathlib import Path

BASE = (DATA_ROOT / 'smoke_test_results')

CELLS = {
    "A": ["django__django", "sympy__sympy", "sphinx-doc__sphinx", "matplotlib__matplotlib",
          "scikit-learn__scikit-learn", "astropy__astropy", "pydata__xarray",
          "pytest-dev__pytest", "pylint-dev__pylint", "psf__requests"],
    "B": ["django__django", "sympy__sympy", "matplotlib__matplotlib",
          "scikit-learn__scikit-learn", "astropy__astropy", "pydata__xarray",
          "pytest-dev__pytest"],
}
MODELS = ["c2llm05", "c2llm7"]

def load(cell, model, repo):
    p = BASE / f"cell{cell}_{model}_c1000" / repo / "smoke_test_django_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

def main():
    rows = []
    for cell in CELLS:
        for model in MODELS:
            n_q = 0
            wt_dense = 0.0
            wt_bm25 = 0.0
            wt_delta = 0.0
            wt_tau = 0.0
            missing = []
            per_repo = []
            for repo in CELLS[cell]:
                s = load(cell, model, repo)
                if s is None:
                    missing.append(repo)
                    continue
                nq = s["n_queries"]
                n_q += nq
                wt_dense += nq * s["ndcg10"]["dense"]
                wt_bm25 += nq * s["ndcg10"]["bm25"]
                wt_delta += nq * s["ndcg10"]["premium_dense_minus_bm25"]
                wt_tau += nq * s["mean_tau_q"]
                per_repo.append((repo, nq,
                                 s["ndcg10"]["dense"],
                                 s["ndcg10"]["bm25"],
                                 s["ndcg10"]["premium_dense_minus_bm25"]))
            if n_q == 0:
                print(f"Cell {cell} x {model}: NO DATA (missing: {missing})")
                continue
            rows.append({
                "cell": cell, "model": model, "n_queries": n_q,
                "dense": wt_dense / n_q,
                "bm25": wt_bm25 / n_q,
                "delta": wt_delta / n_q,
                "tau": wt_tau / n_q,
                "missing": missing,
                "per_repo": per_repo,
            })

    print("\n" + "=" * 90)
    print(f"{'Cell':<6}{'Model':<10}{'n_q':>7}{'Dense':>10}{'BM25':>10}{'Δ':>10}{'τ':>10}")
    print("=" * 90)
    for r in rows:
        print(f"{r['cell']:<6}{r['model']:<10}{r['n_queries']:>7d}"
              f"{r['dense']:>10.4f}{r['bm25']:>10.4f}{r['delta']:>+10.4f}{r['tau']:>10.4f}")
        if r["missing"]:
            print(f"       (missing: {', '.join(r['missing'])})")
    print("=" * 90)

    # scaling uplift 7B - 0.5B, per cell
    print("\nScaling uplift Δ(7B) − Δ(0.5B) per cell:")
    for cell in CELLS:
        d05 = next((r["delta"] for r in rows if r["cell"] == cell and r["model"] == "c2llm05"), None)
        d7 = next((r["delta"] for r in rows if r["cell"] == cell and r["model"] == "c2llm7"), None)
        if d05 is None or d7 is None:
            print(f"  Cell {cell}: incomplete")
            continue
        print(f"  Cell {cell}: {d7:+.4f} − ({d05:+.4f}) = {d7-d05:+.4f}")

    print("\nPer-repo detail:")
    for r in rows:
        print(f"\n[Cell {r['cell']} × {r['model']}]")
        for repo, nq, d, b, delta in r["per_repo"]:
            print(f"  {repo:<30} n={nq:>4d}  dense={d:.4f}  bm25={b:.4f}  Δ={delta:+.4f}")

if __name__ == "__main__":
    main()
