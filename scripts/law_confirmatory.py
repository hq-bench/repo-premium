"""Confirmatory test of the premium-decay law on data never used to select it.

PREREGISTRATION (written 2026-10-03, before any slope on these data was
computed; the only quantities inspected beforehand were the repository lists,
task counts, and the tau recompute guard on three tasks)
---------------------------------------------------------------------------
Predictor (frozen, as in the paper and law_clustered_bootstrap.py):
  tau_top1 = IDF of the rarest query-gold shared token (chunk-level IDF,
  c=1000, first gold file), expressed as the within-(slice, repository)
  percentile among tau>0 queries; (slice, repository) groups with fewer than
  8 tau>0 queries are dropped. Slope = common OLS slope of
  Delta = nDCG@10(dense) - nDCG@10(BM25) on that percentile with slice fixed
  effects. Nothing is refit.

Populations:
  S  = 36-repo matched scale-up (splits_v2 manifest), minus every repository
       that appears in the 19-repository law battery. Slices are
       encoder x arm: {jina-0.5b, qwen3-0.6b} x {curated, raw}.
         S-all     : all four slices          <- PRIMARY test
         S-curated : the two curated slices   (secondary)
         S-raw     : the two raw slices       (secondary)
         per-slice : each of the four         (descriptive)
  D  = DeepSWE, the 68 tasks whose corpus is identical across the jina and
       qwen runs (commit 8d31beb), minus battery overlap. If the frozen
       within-repository rule leaves no repository, D is reported as NOT
       EVALUABLE under the frozen predictor. A clearly labelled DEVIATION
       (percentile taken within slice across repositories) is then reported
       as exploratory only.

Test: one-sided, H1: slope < 0. Uncertainty from a bootstrap that resamples
  repositories, then tasks within repository, carrying every slice
  observation of a task (both arms, both encoders) together, and recomputing
  the percentile inside each resample (law_clustered_bootstrap.rerank).
  B = 4000, seed 20261003.
  SUCCESS iff the upper bound of the two-sided 95% interval is < 0.
  One-sided p = share of bootstrap slopes >= 0.

Magnitude: difference (confirmatory slope - battery slope -0.134) with a 95%
  interval from independent bootstraps of the two (repository-disjoint)
  populations, paired draw by draw. "Compatible" iff that interval covers 0.

Guards (abort if violated): recomputed tau_mean5 equals the stored tau_q for
  every query (|diff| < 1e-9); the battery loader reproduces -0.1341.

AMENDMENT (2026-10-03, still before any slope was computed): the guard
  failed on 20/1440 scale-up rows, all of them vitejs__vite in the jina-0.5b
  run (|diff| 3-4e-4; the qwen3-0.6b vite rows match exactly), i.e. that one
  run's corpus differs by a few chunks from the v2 manifest corpus. The guard
  now tolerates exactly that case (vitejs, jina-0.5b, |diff| < 1e-3) and
  still aborts on anything else; an extra sensitivity population drops
  vitejs entirely. The primary test is unchanged.
---------------------------------------------------------------------------
Usage: conda run -n llm python scripts/law_confirmatory.py
"""
from __future__ import annotations
from paths import DATA_ROOT

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import law_clustered_bootstrap as L  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = DATA_ROOT
SCALE = DATA / "scaleup_pull"
DEEP = DATA / "deepswe_pull"
OUT = DATA / "analysis" / "law_confirmatory.json"
B = 4000
SEED = 20261003
MIN_REPO = 8
BATTERY_SLOPE = -0.1341
ENCODERS = ("jina-0.5b", "qwen3-0.6b")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def build_obs(rows: list[dict], slice_names: list[str], within_repo: bool) -> dict:
    """rows: dicts with slice, repo, task, tau, delta. Mirrors
    law_clustered_bootstrap.within_repo_obs (tau>0, min group size)."""
    groups = defaultdict(list)
    for r in rows:
        if r["tau"] > 0:
            key = (r["slice"], r["repo"] if within_repo else "*")
            groups[key].append(r)
    S, X, Y, R, T, TAU, GRP = [], [], [], [], [], [], []
    dropped = []
    for (sl, g), rs in sorted(groups.items()):
        if len(rs) < MIN_REPO:
            dropped.append((sl, g, len(rs)))
            continue
        x = stats.rankdata([r["tau"] for r in rs]) / len(rs)
        for r, xv in zip(rs, x):
            S.append(slice_names.index(sl)); X.append(xv); Y.append(r["delta"])
            R.append(r["repo"]); T.append(r["task"]); TAU.append(r["tau"])
            GRP.append(f"{sl}|{g}")
    if not Y:
        return {"n": 0, "dropped": dropped}
    _, grp = np.unique(np.array(GRP), return_inverse=True)
    return {"s": np.array(S), "x": np.array(X), "y": np.array(Y),
            "repo": np.array(R), "task": np.array(T), "tau": np.array(TAU),
            "grp": grp, "n": len(Y), "dropped": dropped}


def test(obs: dict, rng, battery_bs: np.ndarray) -> dict:
    if obs["n"] == 0:
        return {"evaluable": False, "dropped_groups": len(obs["dropped"])}
    b = L.fe_slope(obs["s"], obs["x"], obs["y"])
    bs = L.hier_boot_slope(obs, rng, B=B)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    diff = bs - battery_bs[rng.permutation(len(battery_bs))[:len(bs)]]
    dlo, dhi = np.percentile(diff, [2.5, 97.5])
    return {
        "evaluable": True, "n_obs": int(obs["n"]),
        "n_repos": int(len(np.unique(obs["repo"]))),
        "n_tasks": int(len(set(zip(obs["repo"], obs["task"])))),
        "dropped_groups": len(obs["dropped"]),
        "slope": b, "ci95": [float(lo), float(hi)],
        "p_one_sided": float((bs >= 0).mean()),
        "success": bool(hi < 0),
        "diff_vs_battery": b - BATTERY_SLOPE,
        "diff_ci95": [float(dlo), float(dhi)],
        "compatible_with_battery": bool(dlo <= 0 <= dhi),
    }


def scaleup_rows(battery_repos: set[str]) -> tuple[list[dict], dict]:
    manifest = {r["task_id"]: r for r in read_jsonl(SCALE / "splits_v2" / "retrieval_tasks_v2.jsonl")}
    tau = {r["query_id"]: r for r in read_jsonl(SCALE / "tau_top1_scaleup.jsonl")}
    rows, guard_bad, n_q, guard_tolerated = [], 0, 0, 0
    overlap = sorted({m["repo"] for m in manifest.values()
                      if m["repo"].replace("/", "__") in battery_repos})
    for enc in ENCODERS:
        for f in sorted((SCALE / "experiment_v1" / "retrieval" / enc).glob("*/smoke_test_django.jsonl")):
            for r in read_jsonl(f):
                tid, arm = r["query_id"].split("::")
                if tid not in manifest:
                    continue
                t = tau[r["query_id"]]
                n_q += 1
                repo = manifest[tid]["repo"]
                d = abs(t["tau_mean5"] - r["tau_q"])
                if d > 1e-9:
                    if repo == "vitejs/vite" and enc == "jina-0.5b" and d < 1e-3:
                        guard_tolerated += 1
                    else:
                        guard_bad += 1
                if repo in overlap:
                    continue
                rows.append({"slice": f"{enc}|{arm}", "repo": repo, "task": tid,
                             "tau": t["tau_top1"],
                             "delta": r["dense_ndcg10"] - r["bm25_ndcg10"]})
    if guard_bad:
        raise SystemExit(f"GUARD FAILED: {guard_bad}/{n_q} tau_mean5 != stored tau_q")
    return rows, {"overlap_dropped": overlap, "n_queries_checked": n_q,
                  "guard_tolerated_vite_jina": guard_tolerated,
                  "n_tasks_in_manifest": len(manifest)}


def deepswe_rows(battery_repos: set[str]) -> tuple[list[dict], dict]:
    repo_of = dict(l.split() for l in (DEEP / "task_repo.tsv").read_text().splitlines())
    tau = {r["query_id"]: r for r in read_jsonl(DEEP / "tau_top1_deepswe.jsonl")}
    ncorp = {}
    for enc in ENCODERS:
        for f in (DEEP / "retrieval" / enc).glob("*/smoke_test_django_summary.json"):
            ncorp[(enc, f.parent.name)] = json.loads(f.read_text())["n_corpus_files"]
    same = {t for (e, t) in ncorp if e == ENCODERS[0]
            and ncorp.get((ENCODERS[1], t)) == ncorp[(e, t)]}
    rows, bad, nfile_bad = [], 0, 0
    for enc in ENCODERS:
        for t in sorted(same):
            for r in read_jsonl(DEEP / "retrieval" / enc / t / "smoke_test_django.jsonl"):
                tv = tau[r["query_id"]]
                if abs(tv["tau_mean5"] - r["tau_q"]) > 1e-9:
                    bad += 1
                if tv["n_files"] != ncorp[(enc, t)]:
                    nfile_bad += 1
                repo = repo_of[t]
                if repo.replace("/", "__") in battery_repos:
                    continue
                rows.append({"slice": enc, "repo": repo, "task": t,
                             "tau": tv["tau_top1"],
                             "delta": r["dense_ndcg10"] - r["bm25_ndcg10"]})
    if bad or nfile_bad:
        raise SystemExit(f"GUARD FAILED (DeepSWE): tau {bad}, n_files {nfile_bad}")
    return rows, {"n_identical_corpus_tasks": len(same)}


def main() -> None:
    rng = np.random.default_rng(SEED)
    slices = L.load_all()
    names = sorted(n for n in slices if n not in L.CHUNK_DUPES)
    bobs = L.within_repo_obs(slices, names)
    b_bat = L.fe_slope(bobs["s"], bobs["x"], bobs["y"])
    if abs(b_bat - BATTERY_SLOPE) > 5e-4:
        raise SystemExit(f"GUARD FAILED: battery slope {b_bat:+.4f} != {BATTERY_SLOPE}")
    battery_repos = set(np.unique(bobs["repo"]))
    battery_bs = L.hier_boot_slope(bobs, rng, B=B)
    out = {"preregistered": True, "battery_slope": b_bat,
           "battery_ci95": [float(x) for x in np.percentile(battery_bs, [2.5, 97.5])],
           "battery_repos": sorted(battery_repos)}

    rows, meta = scaleup_rows(battery_repos)
    out["scaleup_meta"] = meta
    sl = sorted({r["slice"] for r in rows})
    pops = {"S-all (PRIMARY)": sl,
            "S-curated": [s for s in sl if s.endswith("curated")],
            "S-raw": [s for s in sl if s.endswith("raw")]}
    pops.update({f"slice {s}": [s] for s in sl})
    out["scaleup"] = {}
    for name, keep in pops.items():
        obs = build_obs([r for r in rows if r["slice"] in keep], keep, within_repo=True)
        out["scaleup"][name] = test(obs, rng, battery_bs)
    novite = [r for r in rows if r["repo"] != "vitejs/vite"]
    out["scaleup"]["S-all excl. vitejs (sensitivity)"] = test(
        build_obs(novite, sl, within_repo=True), rng, battery_bs)

    drows, dmeta = deepswe_rows(battery_repos)
    out["deepswe_meta"] = dmeta
    dsl = sorted({r["slice"] for r in drows})
    frozen = build_obs(drows, dsl, within_repo=True)
    out["deepswe_frozen"] = test(frozen, rng, battery_bs)
    out["deepswe_frozen"]["note"] = (
        "within-repository percentile needs >=8 tau>0 queries per (slice, repo)")
    dev = build_obs(drows, dsl, within_repo=False)
    res = test(dev, rng, battery_bs)
    res["DEVIATION"] = ("percentile within slice across repositories; "
                        "exploratory, not the frozen predictor")
    out["deepswe_deviation_pooled_percentile"] = res

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"battery slope {b_bat:+.4f} CI {out['battery_ci95']}")
    print("scale-up overlap dropped:", meta["overlap_dropped"])
    for k, v in {**out["scaleup"], "DeepSWE frozen": out["deepswe_frozen"],
                 "DeepSWE deviation": res}.items():
        if not v["evaluable"]:
            print(f"{k:28s} NOT EVALUABLE (dropped groups {v['dropped_groups']})")
            continue
        print(f"{k:28s} n={v['n_obs']:4d} repos={v['n_repos']:2d} slope={v['slope']:+.4f} "
              f"CI[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] p1={v['p_one_sided']:.4f} "
              f"success={v['success']} diff={v['diff_vs_battery']:+.4f} "
              f"[{v['diff_ci95'][0]:+.4f},{v['diff_ci95'][1]:+.4f}] "
              f"dropped={v['dropped_groups']}")


if __name__ == "__main__":
    main()
