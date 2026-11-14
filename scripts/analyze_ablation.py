#!/usr/bin/env python3
"""Analyse the non-lossy anchor ablation.

The generative arm showed that ADDING lexical anchors removes the dense
premium. This is the mirror: does REMOVING them restore it? The earlier
attempt at that question (the cellC rewrites) was uninterpretable because the
rewrite also compressed the query, so dense lost more than BM25 and the
length confound dominated. Here the query is edited in place: only the K
rarest query tokens that also occur in the gold file are replaced, by
same-length synthetic identifiers, so every other byte is unchanged.

Arms
  guard    original query, re-scored through the same script. Must reproduce
           the cached canonical BM25 to ~1e-12; this validates everything else.
  ablated  bridge severed.
  control  the same number of IDF-matched NON-shared tokens replaced. A
           placebo for "we perturbed the query" that leaves the bridge intact.
  alias    (optional, review #14; anchor_alias_control.py) the SAME tokens as
           `ablated`, renamed to meaning-preserving aliases absent from the
           corpus. Bridge severed exactly as in `ablated`, meaning kept. Its
           BM25 must equal `ablated`'s; the dense contrast alias - ablated
           separates semantic damage (alias ~ guard) from lexical dependence
           (alias ~ ablated).

The quantity of interest is the DOUBLE difference

    (ablated - guard) - (control - guard)  =  ablated - control

which nets out generic perturbation damage and isolates the anchors. Reported
for BM25, for dense, and for the premium (dense - BM25) when a dense stage has
been run; BM25-only otherwise, since the CPU stage is meaningful on its own.

CIs are repository-clustered paired bootstrap, matching analyze_genarm.py.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ARMS = ("guard", "ablated", "control")
ALIAS = "alias"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else float("nan")


def cluster_bootstrap(clusters: list[list[float]], n_boot: int, seed: int):
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
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return point, lo, hi


def load_arm(root: Path, arm: str, cell: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    base = root / f"{arm}_cell{cell}"
    if not base.is_dir():
        return out
    for repo_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        path = repo_dir / "smoke_test_django.jsonl"
        if not path.is_file():
            continue
        for rec in read_jsonl(path):
            rec["repo"] = repo_dir.name
            out[rec["query_id"]] = rec
    return out


def load_cached(canon_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not canon_root.is_dir():
        return out
    for repo_dir in sorted(p for p in canon_root.iterdir() if p.is_dir()):
        path = repo_dir / "smoke_test_django.jsonl"
        if not path.is_file():
            continue
        for rec in read_jsonl(path):
            rec["repo"] = repo_dir.name
            out[rec["query_id"]] = rec
    return out


def paired(rows: dict[str, dict], other: dict[str, dict], field: str,
           unit: str = "repo"):
    """Paired differences rows[field] - other[field], grouped into bootstrap
    clusters by `unit`.

    unit="repo" is the multi-repository default used by the Python cells.
    unit="query" makes each query its own cluster, which is the ordinary
    paired bootstrap; it is the only meaningful choice for a single-repository
    slice such as an R9 language cell, where repo-clustering has one cluster
    and therefore returns a zero-width interval.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for qid, r in rows.items():
        o = other.get(qid)
        if o is None:
            continue
        a, b = r.get(field), o.get(field)
        if a is None or b is None:
            continue
        groups[qid if unit == "query" else r["repo"]].append(a - b)
    return list(groups.values())


def level(rows: dict[str, dict], field: str, unit: str = "repo"):
    groups: dict[str, list[float]] = defaultdict(list)
    for qid, r in rows.items():
        v = r.get(field)
        if v is not None:
            groups[qid if unit == "query" else r["repo"]].append(v)
    return list(groups.values())


def fmt(t) -> str:
    p, lo, hi = t
    return f"{p:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--canon-root", type=Path, required=True,
                    help="Cached canonical run, for the guard check and dense.")
    ap.add_argument("--dense-root", type=Path, default=None,
                    help="GPU-stage tree (ablated/control only). When given, "
                         "those arms are read from here (they carry both BM25 "
                         "and dense), and the guard arm's dense is taken from "
                         "--canon-root, since the guard query IS the original "
                         "query and its BM25 already reproduced the cache.")
    ap.add_argument("--audit-dir", type=Path, default=None)
    ap.add_argument("--alias-audit-dir", type=Path, default=None,
                    help="qrels_cell{X}_alias/ from anchor_alias_control.py; "
                         "enables --alias-complete-only.")
    ap.add_argument("--alias-complete-only", action="store_true",
                    help="Restrict EVERY arm to queries whose tokens were all "
                         "aliased (alias_complete=true), so the alias contrasts "
                         "compare like with like.")
    ap.add_argument("--cell", default="A")
    ap.add_argument("--bootstrap", choices=("repo", "query"), default="repo",
                    help="Bootstrap resampling unit. 'repo' (default) clusters "
                         "by repository, matching the Python cells. 'query' is "
                         "required for single-repository slices (R9 language "
                         "cells), where repo-clustering yields one cluster and "
                         "a zero-width CI.")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    arms = {a: load_arm(args.results_root, a, args.cell) for a in ARMS}
    canon = load_cached(args.canon_root)

    if args.dense_root:
        # Cross-check: the GPU stage re-ran BM25 on the same query text, so its
        # BM25 must equal the CPU stage's exactly. If it does not, the two
        # stages did not see the same queries and nothing below is comparable.
        worst = 0.0
        for a in ("ablated", "control"):
            dense_arm = load_arm(args.dense_root, a, args.cell)
            if not dense_arm:
                continue
            for qid, r in dense_arm.items():
                cpu = arms[a].get(qid)
                if cpu is not None:
                    worst = max(worst, abs(r["bm25_ndcg10"] - cpu["bm25_ndcg10"]))
            arms[a] = dense_arm
        print(f"[check] max |BM25(GPU stage) - BM25(CPU stage)| = {worst:.3e}")
        if worst > 1e-9:
            print("    *** STAGE MISMATCH: arms are not comparable ***")
        # guard dense == cached canonical dense, by construction
        for qid, r in arms["guard"].items():
            c = canon.get(qid)
            if c is not None:
                r["dense_ndcg10"] = c["dense_ndcg10"]
    # Alias arm: GPU stage carries both BM25 and dense; fall back to a
    # CPU-only tree if that is all there is.
    alias = load_arm(args.dense_root, ALIAS, args.cell) if args.dense_root else {}
    if not alias:
        alias = load_arm(args.results_root, ALIAS, args.cell)
    if alias and args.alias_complete_only:
        if not (args.alias_audit_dir and args.alias_audit_dir.is_dir()):
            raise SystemExit("--alias-complete-only needs --alias-audit-dir")
        keep = {r["query_id"] for f in sorted(args.alias_audit_dir.glob("*_audit.jsonl"))
                for r in read_jsonl(f) if r.get("alias_complete")}
        for a in ARMS:
            arms[a] = {q: r for q, r in arms[a].items() if q in keep}
        alias = {q: r for q, r in alias.items() if q in keep}
        print(f"[filter] alias_complete only: {len(keep)} queries kept")
    for a in ARMS:
        print(f"[load] {a}: {len(arms[a])} queries")
    if alias:
        print(f"[load] {ALIAS}: {len(alias)} queries")
    print(f"[load] canonical cache: {len(canon)} queries")

    result: dict[str, Any] = {"cell": args.cell,
                              "bootstrap_unit": args.bootstrap,
                              "n": {a: len(arms[a]) for a in ARMS}}

    # --- guard: the original query must reproduce the cached BM25 exactly
    diffs = [abs(r["bm25_ndcg10"] - canon[q]["bm25_ndcg10"])
             for q, r in arms["guard"].items() if q in canon]
    guard_max = max(diffs) if diffs else float("nan")
    result["guard_max_abs_bm25_diff"] = guard_max
    result["guard_n_compared"] = len(diffs)
    print(f"\n=== guard: max |BM25 - cached| = {guard_max:.3e} over {len(diffs)} queries")
    if diffs and guard_max > 1e-9:
        print("    *** GUARD FAILED: arms are not comparable to the cache ***")

    # --- levels and paired contrasts
    # --skip-dense writes -1.0 SENTINELS, not nulls, so a truthiness test would
    # wrongly report a CPU-only stage as having dense scores.
    def has_dense(rows: dict[str, dict]) -> bool:
        return any((r.get("dense_ndcg10") is not None
                    and r["dense_ndcg10"] > -0.5) for r in rows.values())

    fields = [("bm25_ndcg10", "BM25"), ("tau_q", "tau")]
    dense_available = all(has_dense(arms[a]) for a in ARMS if arms[a])
    if dense_available:
        fields.insert(1, ("dense_ndcg10", "dense"))

    print(f"\n=== arm levels ({args.bootstrap}-clustered mean)")
    for field, label in fields:
        row = []
        for a in ARMS:
            p, lo, hi = cluster_bootstrap(level(arms[a], field, args.bootstrap),
                                          args.n_boot, args.seed)
            row.append(f"{a}={p:.4f}")
            result.setdefault("levels", {}).setdefault(label, {})[a] = [p, lo, hi]
        print(f"  {label:<6} " + "  ".join(row))

    print(f"\n=== paired contrasts vs guard ({args.bootstrap}-clustered bootstrap)")
    for field, label in fields:
        for a in ("ablated", "control"):
            t = cluster_bootstrap(paired(arms[a], arms["guard"], field,
                                         args.bootstrap),
                                  args.n_boot, args.seed)
            result.setdefault("vs_guard", {}).setdefault(label, {})[a] = list(t)
            print(f"  {label:<6} {a:<8} {fmt(t)}")

    print("\n=== double difference: ablated - control (nets out perturbation damage)")
    for field, label in fields:
        t = cluster_bootstrap(paired(arms["ablated"], arms["control"], field,
                                     args.bootstrap),
                              args.n_boot, args.seed)
        result.setdefault("ablated_minus_control", {})[label] = list(t)
        print(f"  {label:<6} {fmt(t)}")

    # --- premium, only if a dense stage exists for the arms
    if dense_available:
        for a in ARMS:
            for r in arms[a].values():
                d, b = r.get("dense_ndcg10"), r.get("bm25_ndcg10")
                if d is not None and b is not None and d > -0.5:
                    r["premium"] = d - b
        print("\n=== premium (dense - BM25)")
        for a in ARMS:
            p, lo, hi = cluster_bootstrap(level(arms[a], "premium", args.bootstrap),
                                          args.n_boot, args.seed)
            result.setdefault("premium", {})[a] = [p, lo, hi]
            print(f"  {a:<8} {p:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        t = cluster_bootstrap(paired(arms["ablated"], arms["control"], "premium",
                                     args.bootstrap),
                              args.n_boot, args.seed)
        result["premium_ablated_minus_control"] = list(t)
        print(f"  ablated - control: {fmt(t)}")
    else:
        print("\n[note] no dense scores in the arms: BM25-only stage. "
              "Premium requires the GPU stage.")

    # --- alias arm (review #14)
    if alias:
        res: dict[str, Any] = {"n": len(alias)}
        # The alias is out of vocabulary, so BM25 must be identical to the
        # nonsense arm query by query; anything else means the two arms did
        # not sever the same bridge and the dense contrast is uninterpretable.
        bdiff = [abs(r["bm25_ndcg10"] - arms["ablated"][q]["bm25_ndcg10"])
                 for q, r in alias.items() if q in arms["ablated"]]
        res["max_abs_bm25_alias_minus_ablated"] = max(bdiff) if bdiff else None
        print(f"\n=== alias arm: max |BM25(alias) - BM25(ablated)| = "
              f"{res['max_abs_bm25_alias_minus_ablated']} over {len(bdiff)} queries")
        if bdiff and max(bdiff) > 1e-9:
            print("    *** ALIAS BRIDGE MISMATCH: alias and ablated differ on BM25 ***")
        alias_dense = has_dense(alias)
        for r in alias.values():
            d, b = r.get("dense_ndcg10"), r.get("bm25_ndcg10")
            if alias_dense and d is not None and b is not None and d > -0.5:
                r["premium"] = d - b
        afields = [("bm25_ndcg10", "BM25")]
        if alias_dense and dense_available:
            afields += [("dense_ndcg10", "dense"), ("premium", "premium")]
        for field, label in afields:
            res.setdefault("level", {})[label] = list(cluster_bootstrap(
                level(alias, field, args.bootstrap), args.n_boot, args.seed))
            for other in ARMS:
                t = cluster_bootstrap(paired(alias, arms[other], field, args.bootstrap),
                                      args.n_boot, args.seed)
                res.setdefault(f"alias_minus_{other}", {})[label] = list(t)
                print(f"  {label:<7} alias - {other:<8} {fmt(t)}")
        if not (alias_dense and dense_available):
            print("  [note] alias arm has no dense scores yet: BM25 only.")
        result[ALIAS] = res

    # --- manipulation audit
    if args.audit_dir and args.audit_dir.is_dir():
        rows = []
        for f in sorted(args.audit_dir.glob("*_audit.jsonl")):
            rows.extend(read_jsonl(f))
        if rows:
            n = len(rows)
            result["audit"] = {
                "queries": n,
                "mean_char_delta": sum(r["chars_ablated"] - r["chars_original"] for r in rows) / n,
                "queries_with_length_change": sum(r["chars_ablated"] != r["chars_original"] for r in rows),
                "mean_shared_before": sum(r["n_shared_before"] for r in rows) / n,
                "mean_shared_after_ablation": sum(r["n_shared_after_ablation"] for r in rows) / n,
                "mean_shared_after_control": sum(r["n_shared_after_control"] for r in rows) / n,
                "mean_tau_mean5_before": sum(r["tau_mean5_before"] for r in rows) / n,
                "mean_tau_mean5_ablated": sum(r["tau_mean5_ablated"] for r in rows) / n,
                "queries_with_unablatable_token": sum(bool(r.get("unablatable_tokens")) for r in rows),
            }
            print("\n=== manipulation audit")
            for k, v in result["audit"].items():
                print(f"  {k:<32} {v}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
