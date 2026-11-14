#!/usr/bin/env python3
"""C4: direct contrast of mining-effect vs date-effect magnitudes (scale-up).

Joint hierarchical bootstrap: resample repositories, then tasks within each
drawn repository *stratified by date window* (so every drawn repository keeps
both its pre and post tasks). On each draw compute, from the same sample:
  M  = mean over tasks of (Delta_raw - Delta_cur)           [mining effect]
  Dv = mean over repos of (mean_post Delta_v - mean_pre Delta_v), v in {cur, raw}
and the contrasts |M| - |Dv| and Dv / M. Point estimates reproduce
analyze_scaleup.py (M=+0.0893, D_cur=-0.0658 for jina-0.5b).
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def load(root: Path, manifest: Path):
    meta = {r["task_id"]: r for r in map(json.loads, manifest.read_text().splitlines())}
    by = defaultdict(lambda: {"pre": [], "post": []})
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        f = d / "smoke_test_django.jsonl"
        if not f.is_file() or d.name not in meta:
            continue
        v = {r["query_id"].rsplit("::", 1)[-1]: r for r in map(json.loads, f.read_text().splitlines())}
        if {"curated", "raw"} - set(v):
            continue
        dc = v["curated"]["dense_ndcg10"] - v["curated"]["bm25_ndcg10"]
        dr = v["raw"]["dense_ndcg10"] - v["raw"]["bm25_ndcg10"]
        info = meta[d.name]
        by[info["repo"]][info["date_window"]].append((dc, dr))
    return {k: {w: np.array(x) for w, x in v.items()} for k, v in by.items()}


def stats(sample):
    allt = np.concatenate([np.concatenate([s["pre"], s["post"]]) for s in sample])
    M = float(np.mean(allt[:, 1] - allt[:, 0]))
    D = {}
    for j, name in ((0, "cur"), (1, "raw")):
        diffs = [s["post"][:, j].mean() - s["pre"][:, j].mean()
                 for s in sample if len(s["pre"]) and len(s["post"])]
        D[name] = float(np.mean(diffs))
    return M, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    by = load(a.root, a.manifest)
    repos = sorted(by)
    M, D = stats([by[r] for r in repos])
    rng = np.random.default_rng(a.seed)
    bs = defaultdict(list)
    for _ in range(a.n_boot):
        samp = []
        for r in rng.choice(repos, len(repos)):
            s = by[r]
            samp.append({w: s[w][rng.integers(0, len(s[w]), len(s[w]))] if len(s[w]) else s[w]
                         for w in ("pre", "post")})
        m, d = stats(samp)
        bs["M"].append(m)
        for v in ("cur", "raw"):
            bs[f"D_{v}"].append(d[v])
            bs[f"absM_minus_absD_{v}"].append(abs(m) - abs(d[v]))
            bs[f"ratio_D_{v}_over_M"].append(d[v] / m if m else np.nan)
            bs[f"P_absD_ge_absM_{v}"].append(float(abs(d[v]) >= abs(m)))
    pts = {"M": M, "D_cur": D["cur"], "D_raw": D["raw"],
           "absM_minus_absD_cur": abs(M) - abs(D["cur"]), "absM_minus_absD_raw": abs(M) - abs(D["raw"]),
           "ratio_D_cur_over_M": D["cur"] / M, "ratio_D_raw_over_M": D["raw"] / M}
    out = {"n_repos": len(repos), "n_tasks": int(sum(len(v["pre"]) + len(v["post"]) for v in by.values()))}
    for k, x in bs.items():
        x = np.array(x)
        if k.startswith("P_"):
            out[k] = float(np.nanmean(x))
        else:
            out[k] = {"point": pts[k], "ci": [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]}
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
