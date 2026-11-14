"""tau_stratified_agent — the preregistered Q1 test at the agent boundary.

Q1 predicts that low-tau queries benefit disproportionately from the dense /
router wrappers over BM25.  sec:agent registered the quartile contrast when
the affordance sample was n=30 (~7 per bin, nothing resolvable); the n=96
sample supports it at ~24 per bin from trajectories already on disk.

This script computes nothing new from the agents: it reads the recorded
per-query outcomes of the four fixed-LLM arms, attaches the deployment-time
tau-hat of each query, and reports the wrapper contrast by tau-hat quartile.

tau-hat is `embedding_premium.Router.tau_query` -- top-5 mean IDF of the
boilerplate-stripped query against the *same* c=1000 chunks the wrappers
retrieved over (`indices/<repo>/chunks.jsonl`), so the router sees exactly
the token distribution sec:tau measures offline.

Usage (where the run outputs live):
    python3 tau_stratified_agent.py \
        --sample      $D/agent_eval_100.jsonl \
        --indices     $D/agent_eval_wrapper/indices \
        --runs        $D/agent_eval_runs_n100 \
        --out         $D/analysis/tau_stratified_agent.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from embedding_premium import Router  # noqa: E402

ARMS = ["raw_shell_qwen122b", "wrapper_grep", "wrapper_dense", "wrapper_router"]
# The contrast Q1 names, plus the two that share its logic.
CONTRASTS = [
    ("wrapper_dense", "wrapper_grep"),
    ("wrapper_router", "wrapper_grep"),
    ("raw_shell_qwen122b", "wrapper_grep"),
]
METRICS = ["exact_match", "any_hit", "precision", "recall"]
N_BOOT = 10_000
SEED = 0


def repo_slug(repo: str) -> str:
    return repo.replace("/", "__")


def load_sample(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["query_id"]] = rec
    return out


def load_arm(runs: Path, arm: str) -> dict[str, dict]:
    """Per-query outcomes for one arm, keyed by query_id."""
    out = {}
    for p in sorted((runs / arm).glob("*.result.json")):
        rec = json.loads(p.read_text())
        out[rec["query_id"]] = {
            "exact_match": float(bool(rec["exact_match"])),
            "any_hit": float(rec["n_hit"] > 0),
            "precision": float(rec["precision"]),
            "recall": float(rec["recall"]),
            "timed_out": bool(rec.get("timed_out", False)),
            "n_search_bypass": int(rec.get("search_audit", {}).get("n_search_bypass", 0)),
        }
    return out


def build_routers(indices: Path, repos: set[str]) -> dict[str, Router]:
    routers = {}
    for slug in sorted(repos):
        chunk_file = indices / slug / "chunks.jsonl"
        if not chunk_file.exists():
            raise FileNotFoundError(f"no index for {slug}: {chunk_file}")
        chunks = []
        with chunk_file.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line)["text"])
        routers[slug] = Router.fit(chunks)
    return routers


def paired_bootstrap(deltas: list[float], n_boot: int = N_BOOT,
                     seed: int = SEED) -> tuple[float, float, float]:
    """Mean of a paired per-query difference with a percentile CI."""
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(deltas) / n
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return mean, lo, hi


def quartile_bounds(values: list[float]) -> list[float]:
    """Interior quartile cut points of the observed sample."""
    xs = sorted(values)
    n = len(xs)
    return [xs[int(q * n)] for q in (0.25, 0.50, 0.75)]


def assign_quartile(x: float, cuts: list[float]) -> int:
    for i, c in enumerate(cuts):
        if x < c:
            return i
    return 3


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--indices", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sample = load_sample(args.sample)
    arms = {a: load_arm(args.runs, a) for a in ARMS}

    # Paired analysis: only queries every arm actually produced.
    qids = sorted(set.intersection(*(set(v) for v in arms.values())) & set(sample))
    print(f"queries in sample: {len(sample)}; paired across all arms: {len(qids)}")

    routers = build_routers(args.indices, {repo_slug(sample[q]["repo"]) for q in qids})
    tau = {q: routers[repo_slug(sample[q]["repo"])].tau_query(sample[q]["problem_statement"])
           for q in qids}

    cuts = quartile_bounds([tau[q] for q in qids])
    quart = {q: assign_quartile(tau[q], cuts) for q in qids}
    print(f"tau-hat quartile cuts: {[round(c, 3) for c in cuts]}")

    out: dict = {
        "n_paired": len(qids),
        "tau_cuts": cuts,
        "n_boot": N_BOOT,
        "seed": SEED,
        "per_query": [
            {"query_id": q, "repo": sample[q]["repo"], "tau_hat": tau[q],
             "quartile": quart[q] + 1,
             **{a: arms[a][q]["exact_match"] for a in ARMS}}
            for q in qids
        ],
        "pooled": {}, "by_quartile": {}, "correlation": {},
    }

    # --- pooled arm means -------------------------------------------------
    for a in ARMS:
        out["pooled"][a] = {m: sum(arms[a][q][m] for q in qids) / len(qids) for m in METRICS}

    # --- the preregistered contrast, pooled and by quartile ---------------
    for hi_arm, lo_arm in CONTRASTS:
        key = f"{hi_arm}_minus_{lo_arm}"
        d_all = [arms[hi_arm][q]["exact_match"] - arms[lo_arm][q]["exact_match"] for q in qids]
        m, lo, hi = paired_bootstrap(d_all)
        out["pooled"][key] = {"exact_match": {"mean": m, "ci": [lo, hi], "n": len(qids)}}

        rows = {}
        for qi in range(4):
            sub = [q for q in qids if quart[q] == qi]
            row = {"n": len(sub),
                   "tau_mean": sum(tau[q] for q in sub) / len(sub) if sub else 0.0}
            for m_name in ("exact_match", "any_hit"):
                d = [arms[hi_arm][q][m_name] - arms[lo_arm][q][m_name] for q in sub]
                mm, l, h = paired_bootstrap(d, seed=SEED + qi)
                row[m_name] = {"mean": mm, "ci": [l, h],
                               hi_arm: sum(arms[hi_arm][q][m_name] for q in sub) / len(sub) if sub else 0.0,
                               lo_arm: sum(arms[lo_arm][q][m_name] for q in sub) / len(sub) if sub else 0.0}
            rows[f"Q{qi + 1}"] = row
        out["by_quartile"][key] = rows

        # Q1 predicts a bottom-vs-top gap: Q1 delta minus Q4 delta.
        # Run it on both metrics -- reaching a gold file at all and
        # reproducing the gold set exactly are different questions.
        out["by_quartile"][key]["bottom_minus_top"] = {}
        out["correlation"][key] = {}
        for mi, m_name in enumerate(("exact_match", "any_hit")):
            b = [arms[hi_arm][q][m_name] - arms[lo_arm][q][m_name]
                 for q in qids if quart[q] == 0]
            t = [arms[hi_arm][q][m_name] - arms[lo_arm][q][m_name]
                 for q in qids if quart[q] == 3]
            rng = random.Random(SEED + 99 + mi)
            diffs = []
            for _ in range(N_BOOT):
                bb = sum(b[rng.randrange(len(b))] for _ in range(len(b))) / len(b)
                tt = sum(t[rng.randrange(len(t))] for _ in range(len(t))) / len(t)
                diffs.append(bb - tt)
            diffs.sort()
            out["by_quartile"][key]["bottom_minus_top"][m_name] = {
                "mean": (sum(b) / len(b)) - (sum(t) / len(t)),
                "ci": [diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT) - 1]],
            }
            # Continuous form of the same prediction: no binning.
            d_m = [arms[hi_arm][q][m_name] - arms[lo_arm][q][m_name] for q in qids]
            out["correlation"][key][m_name] = {
                "spearman_tau_vs_delta": spearman([tau[q] for q in qids], d_m)
            }

    # --- audit: the numbers only mean something on a clean run ------------
    out["audit"] = {
        a: {"n_bypass_nonzero": sum(1 for q in qids if arms[a][q]["n_search_bypass"] > 0),
            "n_timeout": sum(1 for q in qids if arms[a][q]["timed_out"])}
        for a in ARMS
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    # --- console summary --------------------------------------------------
    print("\npooled exact-match:", {a: round(out["pooled"][a]["exact_match"], 4) for a in ARMS})
    for hi_arm, lo_arm in CONTRASTS:
        key = f"{hi_arm}_minus_{lo_arm}"
        print(f"\n=== {key} (exact-match) ===")
        p = out["pooled"][key]["exact_match"]
        print(f"  pooled {p['mean']:+.4f} [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]")
        for m_name in ("exact_match", "any_hit"):
            print(f"  -- {m_name} --")
            for qi in range(4):
                r = out["by_quartile"][key][f"Q{qi + 1}"]
                e = r[m_name]
                print(f"  Q{qi+1} (n={r['n']:2d}, tau={r['tau_mean']:.2f}): "
                      f"{e['mean']:+.4f} [{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}]   "
                      f"{lo_arm}={e[lo_arm]:.3f} -> {hi_arm}={e[hi_arm]:.3f}")
            bt = out["by_quartile"][key]["bottom_minus_top"][m_name]
            print(f"  Q1-Q4 {bt['mean']:+.4f} [{bt['ci'][0]:+.4f}, {bt['ci'][1]:+.4f}]")
            print("  spearman(tau, delta) = "
                  f"{out['correlation'][key][m_name]['spearman_tau_vs_delta']:+.3f}")
    print("\naudit:", json.dumps(out["audit"]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
