#!/usr/bin/env python3
"""Equivalence / non-inferiority tests (TOST) for the paper's null-type claims.

Review #6, #7, #13: "attenuates", "does not rise", "at least as good" were
being read off intervals that merely include zero. This script tests each such
claim against a declared smallest effect size of interest (SESOI):

    delta = 0.05  (nDCG@10; applied unchanged to exact-match proportions)

  equivalence      H0: |theta| >= delta.  Pass iff the 90% CI lies inside
                   (-delta, +delta)  (two one-sided tests at alpha = .05).
  non-inferiority  H0: theta >= +delta (the claim is "theta does not exceed
                   delta"). Pass iff the upper 90% bound < +delta.

Every claim is re-estimated with the SAME bootstrap the paper used for it
(same resampling unit, seed, draw count and quantile indexing). The published
95% CI is reproduced first as a loader check; the 90% CI and a bootstrap TOST
p-value come from the same draws. Holm's step-down correction is applied
across the whole family of intervention claims.

Inputs: local caches under ${REPO_PREMIUM_DATA:-data}/smoke_test_results/ plus the
cluster pulls under ${REPO_PREMIUM_DATA:-data}/remote_pull/ (genarm, ablation,
ablation_r9, agent_n100 *.result.json).
"""
from __future__ import annotations
from paths import DATA_ROOT

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

DELTA = 0.05
D = DATA_ROOT
SMOKE = D / "smoke_test_results"
CANON = SMOKE / "canon202607"
PULL = D / "remote_pull"


# --------------------------------------------------------------------------- io
def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load_run(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not root.is_dir():
        raise FileNotFoundError(root)
    for repo_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        f = repo_dir / "smoke_test_django.jsonl"
        if f.is_file():
            for r in read_jsonl(f):
                r["repo"] = repo_dir.name
                out[r["query_id"]] = r
    return out


# --------------------------------------------------------- the paper's bootstraps
# Each returns sorted draws; quantile extraction mirrors the original script.
def draws_cluster(clusters: list[list[float]], n_boot: int, seed: int) -> list[float]:
    """analyze_genarm.py / analyze_ablation.py: resample clusters, keep members."""
    rng = random.Random(seed)
    n = len(clusters)
    out = []
    for _ in range(n_boot):
        s: list[float] = []
        for _ in range(n):
            s.extend(clusters[rng.randrange(n)])
        if s:
            out.append(sum(s) / len(s))
    out.sort()
    return out


def q_cluster(d: list[float], a: float) -> tuple[float, float]:
    return d[int(a / 2 * len(d))], d[min(len(d) - 1, int((1 - a / 2) * len(d)))]


def draws_two_stage(by_repo: dict[str, list[float]], n_boot: int, seed: int) -> list[float]:
    """inverse_intervention.py: resample repositories, then queries within."""
    repos = list(by_repo)
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        s: list[float] = []
        for _ in range(len(repos)):
            v = by_repo[repos[rng.randrange(len(repos))]]
            s.extend(v[rng.randrange(len(v))] for _ in v)
        out.append(sum(s) / len(s))
    out.sort()
    return out


def q_two_stage(d: list[float], a: float) -> tuple[float, float]:
    return d[int(a / 2 * len(d))], d[int((1 - a / 2) * len(d))]


def draws_paired(deltas: list[float], n_boot: int, seed: int) -> list[float]:
    """tau_stratified_agent.py: ordinary paired bootstrap over queries."""
    n = len(deltas)
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        out.append(s / n)
    out.sort()
    return out


def q_paired(d: list[float], a: float) -> tuple[float, float]:
    return d[int(a / 2 * len(d))], d[int((1 - a / 2) * len(d)) - 1]


# ----------------------------------------------------------------------- claims
class Claim:
    def __init__(self, cid, text, tex_line, kind, point, draws, qfn, published):
        self.cid, self.text, self.tex_line, self.kind = cid, text, tex_line, kind
        self.point, self.draws, self.qfn, self.published = point, draws, qfn, published

    def evaluate(self) -> dict:
        lo95, hi95 = self.qfn(self.draws, 0.05)
        lo90, hi90 = self.qfn(self.draws, 0.10)
        n = len(self.draws)
        p_upper = sum(1 for x in self.draws if x >= DELTA) / n   # H0: theta >= +delta
        p_lower = sum(1 for x in self.draws if x <= -DELTA) / n  # H0: theta <= -delta
        if self.kind == "equivalence":
            p = max(p_upper, p_lower)
            passed = -DELTA < lo90 and hi90 < DELTA
        else:  # non-inferiority: claim theta < +delta
            p = p_upper
            passed = hi90 < DELTA
        pub = self.published
        repro = (pub is None) or (abs(round(self.point, 4) - pub[0]) <= 6e-4
                                  and abs(lo95 - pub[1]) <= 6e-4 and abs(hi95 - pub[2]) <= 6e-4)
        return {"id": self.cid, "claim": self.text, "tex_line": self.tex_line,
                "kind": self.kind, "estimate": self.point,
                "ci95": [lo95, hi95], "ci90": [lo90, hi90],
                "published95": pub, "reproduces_published": repro,
                "p_tost": p, "pass_unadjusted": passed}


def mean(xs):
    return sum(xs) / len(xs)


def genarm_claims(n_boot: int) -> list[Claim]:
    seed = 20260813
    cells = {
        # cell: (canonical dense dir, strongest-encoder dense dir, published canon, published best)
        "A": (CANON / "jina05_cellA_c1000", CANON / "jina15_cellA_c1000",
              (-0.0202, -0.0407, 0.0114), (-0.0173, -0.0448, -0.0034)),
        "B": (CANON / "jina05_cellB_c1000", CANON / "jina15_cellB_c1000",
              (0.0304, -0.0459, 0.1283), (0.0665, -0.0068, 0.1556)),
        "RA": (SMOKE / "r8_robust_jina-0.5b_c1000", SMOKE / "r8_robust_jina-0.5b_c1000",
               (0.0196, -0.0425, 0.0909), (0.0196, -0.0425, 0.0909)),
        "RB": (SMOKE / "r8_2022_jina-0.5b_c1000", SMOKE / "r8_2022_qwen3-4b_c1000",
               (-0.0067, -0.0601, 0.0479), (0.0295, -0.0175, 0.0701)),
    }
    line = {"canon": 1302, "best": 1336}
    out = []
    for cell, (canon_dir, best_dir, pub_c, pub_b) in cells.items():
        exp = load_run(PULL / "genarm/results" / f"expanded_cell{cell}")
        ctrl = load_run(PULL / "genarm/results" / f"control_cell{cell}")
        gen = load_run(PULL / "genarm/results" / f"generated_cell{cell}")
        for tag, ddir, pub in (("canon", canon_dir, pub_c), ("best", best_dir, pub_b)):
            if tag == "best" and ddir == canon_dir:
                continue  # canonical encoder is already strongest (Cell RA): same claim
            dense = load_run(ddir)
            common = sorted(set(exp) & set(ctrl) & set(gen) & set(dense))
            groups: dict[str, list[float]] = defaultdict(list)
            for q in common:
                groups[ctrl[q]["repo"]].append(dense[q]["dense_ndcg10"] - exp[q]["bm25_ndcg10"])
            vals = [v for g in groups.values() for v in g]
            enc = ddir.name.split("_c1000")[0]
            out.append(Claim(
                f"genarm_{tag}_{cell}",
                f"post-expansion premium, Cell {cell}, vs {enc} (n={len(vals)})",
                line[tag], "equivalence", mean(vals),
                draws_cluster(list(groups.values()), n_boot, seed), q_cluster, pub))
    return out


def inverse_claims(n_boot: int) -> list[Claim]:
    seed = 20260821
    enc_map = {"jina-0.5b": "jina05", "jina-1.5b": "jina15",
               "Qwen3-0.6B": "qwen06", "Qwen3-4B": "qwen4"}
    pooled: dict[str, list[float]] = defaultdict(list)
    out = []
    for cname, prefix in enc_map.items():
        cc = load_run(SMOKE / f"cellC_{cname}_c1000")
        src = {}
        for cell in ("cellA", "cellB"):
            src.update(load_run(CANON / f"{prefix}_{cell}_c1000"))
        by_repo: dict[str, list[float]] = defaultdict(list)
        for qid, r in cc.items():
            s = src.get(qid[:-2] if qid.endswith("_c") else qid)
            if s is None:
                continue
            eff = (r["dense_ndcg10"] - r["bm25_ndcg10"]) - (s["dense_ndcg10"] - s["bm25_ndcg10"])
            by_repo[r["repo"]].append(eff)
            pooled[r["repo"]].append(eff)
        if cname == "jina-0.5b":
            vals = [v for g in by_repo.values() for v in g]
            out.append(Claim("inverse_canon", f"Cell C rewrite premium change, jina-0.5b (n={len(vals)})",
                             1365, "noninferiority", mean(vals),
                             draws_two_stage(by_repo, n_boot, seed), q_two_stage,
                             (-0.039, -0.165, 0.091)))
    vals = [v for g in pooled.values() for v in g]
    out.append(Claim("inverse_pooled", f"Cell C premium change pooled over 4 encoders (n={len(vals)})",
                     1366, "noninferiority", mean(vals),
                     draws_two_stage(pooled, n_boot, seed), q_two_stage,
                     (-0.048, -0.134, 0.047)))
    return out


def ablation_claims(n_boot: int) -> list[Claim]:
    seed = 0
    specs = [
        ("abl_A", "Cell A", PULL / "ablation/results_dense/ablated_cellA",
         PULL / "ablation/results_dense/control_cellA", "repo", 1393,
         (0.0090, -0.0235, 0.0330)),
        ("abl_B", "Cell B", PULL / "ablation/results_dense/ablated_cellB",
         PULL / "ablation/results_dense/control_cellB", "repo", 1415,
         (-0.0412, -0.1646, 0.0203)),
        ("abl_rust", "tokio (Rust)", PULL / "ablation_r9/results/rust_dense/ablated",
         PULL / "ablation_r9/results/rust_dense/control", "query", 1430,
         (0.0353, -0.0215, 0.0926)),
        ("abl_ts", "TypeScript", PULL / "ablation_r9/results/ts_dense/ablated",
         PULL / "ablation_r9/results/ts_dense/control", "query", 1430,
         (0.0493, 0.0020, 0.0990)),
    ]
    out = []
    for cid, label, adir, cdir, unit, line, pub in specs:
        ab, ct = load_run(adir), load_run(cdir)
        groups: dict[str, list[float]] = defaultdict(list)
        for q, r in ab.items():
            c = ct.get(q)
            if c is None or r["dense_ndcg10"] < -0.5 or c["dense_ndcg10"] < -0.5:
                continue
            d = (r["dense_ndcg10"] - r["bm25_ndcg10"]) - (c["dense_ndcg10"] - c["bm25_ndcg10"])
            groups[q if unit == "query" else r["repo"]].append(d)
        vals = [v for g in groups.values() for v in g]
        out.append(Claim(cid, f"anchor ablation premium change (ablated-placebo), {label} (n={len(vals)})",
                         line, "noninferiority", mean(vals),
                         draws_cluster(list(groups.values()), n_boot, seed), q_cluster, pub))
    return out


def agent_claims(n_boot: int) -> list[Claim]:
    runs = PULL / "agent_n100"

    def arm(name):
        o = {}
        for p in sorted((runs / name).glob("*.result.json")):
            r = json.loads(p.read_text())
            o[r["query_id"]] = float(bool(r["exact_match"]))
        return o

    raw = arm("raw_shell_qwen122b")
    pubs = {"wrapper_grep": (-0.031, -0.104, 0.042), "wrapper_dense": (-0.010, -0.094, 0.073),
            "wrapper_router": (-0.042, -0.115, 0.031)}
    out = []
    for w, pub in pubs.items():
        wa = arm(w)
        qids = sorted(set(raw) & set(wa))
        d = [raw[q] - wa[q] for q in qids]
        out.append(Claim(f"agent_{w}", f"exact-match, raw shell minus {w} (n={len(qids)})",
                         3553, "noninferiority", mean(d), draws_paired(d, n_boot, 0), q_paired, pub))
    return out


def holm(results: list[dict], key: str = "holm") -> None:
    order = sorted(range(len(results)), key=lambda i: results[i]["p_tost"])
    m = len(results)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * results[i]["p_tost"])
        running = max(running, adj)
        results[i][f"p_{key}"] = running
        results[i][f"pass_{key}"] = running < 0.05


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", type=Path, default=D / "analysis/tost_claims.json")
    args = ap.parse_args()

    claims: list[Claim] = []
    for fn in (genarm_claims, inverse_claims, ablation_claims, agent_claims):
        claims.extend(fn(args.n_boot))
    res = [c.evaluate() for c in claims]
    for r in res:
        r["family"] = r["id"].split("_")[0]
    holm(res)  # one family: every intervention claim
    for fam in sorted({r["family"] for r in res}):
        holm([r for r in res if r["family"] == fam], key="holm_within")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"delta": DELTA, "alpha": 0.05,
                                    "multiplicity": "Holm over all claims (pass_holm); Holm within intervention (pass_holm_within)", "claims": res}, indent=2))
    for r in res:
        print(f"{r['id']:22s} {r['kind'][:5]} est={r['estimate']:+.4f} "
              f"95=[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] pub={r['published95']} "
              f"repro={'Y' if r['reproduces_published'] else 'N'} "
              f"90=[{r['ci90'][0]:+.4f},{r['ci90'][1]:+.4f}] p={r['p_tost']:.4f} "
              f"pass={r['pass_unadjusted']} holm={r['pass_holm']} within={r['pass_holm_within']}")


if __name__ == "__main__":
    main()
