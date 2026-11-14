"""Repository-clustered reanalysis of the premium-decay law (review #1, #12).

law_deepdive.py reports the pooled common slope with a *slice*-cluster
bootstrap and an ordinary pooled F-test. Both treat the 38 slices as
independent, but the thirteen encoders on Cur-Pre/Raw-Post share queries
(and the BM25 arm), and Desc rewrites are derived from Cur-Pre/Raw-Post
queries. This script keeps that dependence intact:

  * cluster = repository (org__repo), shared across every slice;
  * within a resampled repository, *tasks* are resampled, where a task is
    the base query id (Desc's trailing "_c" stripped, so a rewrite travels
    with its source query) and every slice observation of that task is
    carried along together;
  * the slope test is a CR1 cluster-robust t-test (G-1 df) plus a
    restricted wild-cluster bootstrap (Webb 6-point weights), both
    clustered on repository.

It also recomputes the cross-cell premium gap (Raw-Post minus Cur-Pre,
jina-code-0.5b) with a repository-block bootstrap (#12), and the
transport result restricted to slices not used to select MaxIDF (#2).

Inputs are the durable local mirror (${REPO_PREMIUM_DATA:-data}/smoke_test_results);
the ${REPO_PREMIUM_DATA:-data}/coreb_pull and ${REPO_PREMIUM_DATA:-data}/coreb_law paths law_mining.py expects no
longer exist.

Usage:
  conda run -n llm python scripts/law_clustered_bootstrap.py
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
from law_mining import load_slice, pct_rank  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SMOKE = DATA_ROOT / "smoke_test_results"
OUT = DATA_ROOT / "analysis" / "law_clustered_bootstrap.json"
N_BOOT = 4000
VARIANT = "tau_top1"
CHUNK_DUPES = {"py_cur_jina05_c1500", "py_cur_jina05_c2000", "py_cur_jina05_c3000"}


def slice_table() -> dict[str, tuple[Path, Path]]:
    canon = SMOKE / "canon202607"
    tv_ab = {"A": SMOKE / "tau_variants_canon202607" / "cellA",
             "B": SMOKE / "tau_variants_canon202607" / "cellB"}
    tvl = SMOKE / "tau_variants_law"
    s: dict[str, tuple[Path, Path]] = {}
    s["py_cur_jina05"] = (canon / "jina05_cellA_c1000", tv_ab["A"])
    s["py_raw_jina05"] = (canon / "jina05_cellB_c1000", tv_ab["B"])
    s["py_r23_jina05"] = (SMOKE / "r8_robust_jina-0.5b_c1000", tvl / "r5")
    s["py_r22_jina05"] = (SMOKE / "r8_2022_jina-0.5b_c1000", tvl / "r8_2022")
    s["py_ood_jina05"] = (SMOKE / "heldout_jina05_c1000", tvl / "heldout")
    s["py_desc_jina05"] = (SMOKE / "cellC_jina-0.5b_c1000", tvl / "cellC")
    for c in (1500, 2000, 3000):
        s[f"py_cur_jina05_c{c}"] = (canon / f"jina05_cellA_c{c}", tv_ab["A"])
    s["js_vuejs_jina05"] = (SMOKE / "r9_vuejs_2023_jina-0.5b_c1000", tvl / "r9_vuejs")
    s["ts_jina05"] = (SMOKE / "r9b_ts_c1000", tvl / "r9b_ts")
    s["go_jina05"] = (SMOKE / "r9c_go_c1000", tvl / "r9c_go")
    s["rust_jina05"] = (SMOKE / "r9d_rust_c1000", tvl / "r9d_rust")
    for tag in ("jina15", "qwen06", "qwen4", "qwen8", "c2llm05", "c2llm7",
                "f2llm06", "f2llm4", "e5base", "e5large", "gteq15", "gteq7"):
        s[f"py_cur_{tag}"] = (canon / f"{tag}_cellA_c1000", tv_ab["A"])
        s[f"py_raw_{tag}"] = (canon / f"{tag}_cellB_c1000", tv_ab["B"])
    s["py_r22_qwen4"] = (SMOKE / "r8_2022_qwen3-4b_c1000", tvl / "r8_2022")
    s["py_desc_jina15"] = (SMOKE / "cellC_jina-1.5b_c1000", tvl / "cellC")
    s["py_desc_qwen06"] = (SMOKE / "cellC_Qwen3-0.6B_c1000", tvl / "cellC")
    s["py_desc_qwen4"] = (SMOKE / "cellC_Qwen3-4B_c1000", tvl / "cellC")
    return s


def repo_of(qid: str) -> str:
    q = qid[:-2] if qid.endswith("_c") else qid
    return q.rsplit("-pr-", 1)[0] if "-pr-" in q else q.rsplit("-", 1)[0]


def task_of(qid: str) -> str:
    return qid[:-2] if qid.endswith("_c") else qid


def load_all() -> dict[str, list[dict]]:
    slices = {}
    for name, (rdir, tdir) in slice_table().items():
        if not rdir.exists() or not tdir.exists():
            print(f"[skip] {name}: missing {rdir if not rdir.exists() else tdir}")
            continue
        rows = load_slice(rdir, tdir)
        if len(rows) >= 20:
            slices[name] = rows
    return slices


def within_repo_obs(slices: dict, names: list[str], min_repo: int = 8) -> dict:
    """Long table of tau>0 observations with within-(slice, repo) percentile,
    mirroring law_deepdive.xy_within_repo exactly (same drop rule)."""
    S, X, Y, R, T, TAU, GRP = [], [], [], [], [], [], []
    for si, name in enumerate(names):
        by_repo = defaultdict(list)
        for r in slices[name]:
            if r[VARIANT] > 0:
                by_repo[repo_of(r["query_id"])].append(r)
        for repo, rows in by_repo.items():
            if len(rows) < min_repo:
                continue
            x = pct_rank(np.array([r[VARIANT] for r in rows]))
            for r, xv in zip(rows, x):
                S.append(si); X.append(xv); Y.append(r["delta"])
                R.append(repo); T.append(task_of(r["query_id"]))
                TAU.append(r[VARIANT]); GRP.append(f"{si}|{repo}")
    _, grp = np.unique(np.array(GRP), return_inverse=True)
    return {"s": np.array(S), "x": np.array(X), "y": np.array(Y),
            "repo": np.array(R), "task": np.array(T), "k": len(names),
            "tau": np.array(TAU), "grp": grp}


def rerank(obs: dict, ix: np.ndarray) -> np.ndarray:
    """Recompute the within-(slice, repo) percentile on a resample. The
    percentile is a rank statistic: holding it at full-sample values while
    resampling tasks breaks its within-group uniformity and lets
    between-repository offsets leak into the slope (inflates the CI ~3x)."""
    g = obs["grp"][ix]; t = obs["tau"][ix]
    x = np.empty(len(ix))
    order = np.argsort(g, kind="stable")
    gs = g[order]
    cuts = np.flatnonzero(np.diff(gs)) + 1
    for blk in np.split(order, cuts):
        x[blk] = stats.rankdata(t[blk]) / len(blk)
    return x


def fe_slope(s: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Common slope with slice fixed effects = within-slice demeaned OLS."""
    xd = x.copy(); yd = y.copy()
    for g in np.unique(s):
        m = s == g
        xd[m] -= x[m].mean(); yd[m] -= y[m].mean()
    return float((xd @ yd) / (xd @ xd))


def demean(s, v):
    out = v.astype(float).copy()
    for g in np.unique(s):
        m = s == g
        out[m] -= v[m].mean()
    return out


def cr1_t(s, x, y, cl) -> tuple[float, float, float, int]:
    """FE slope, CR1 cluster-robust SE, t, and number of clusters."""
    xd, yd = demean(s, x), demean(s, y)
    b = (xd @ yd) / (xd @ xd)
    e = yd - b * xd
    clusters = np.unique(cl)
    G = len(clusters)
    meat = sum((xd[cl == c] @ e[cl == c]) ** 2 for c in clusters)
    n, k = len(y), len(np.unique(s)) + 1
    adj = G / (G - 1) * (n - 1) / (n - k)
    se = np.sqrt(adj * meat) / (xd @ xd)
    return float(b), float(se), float(b / se), G


def wild_cluster_p(s, x, y, cl, rng, B=9999) -> float:
    """Restricted (null b=0) wild cluster bootstrap-t, Webb 6-point weights."""
    xd, yd = demean(s, x), demean(s, y)
    _, _, t_obs, _ = cr1_t(s, x, y, cl)
    clusters, inv = np.unique(cl, return_inverse=True)
    G = len(clusters)
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5),
                     np.sqrt(0.5), 1, np.sqrt(1.5)])
    # under H0 the restricted residual is just yd (FE-only fit)
    tb = np.empty(B)
    xx = xd @ xd
    n, k = len(y), len(np.unique(s)) + 1
    adj = G / (G - 1) * (n - 1) / (n - k)
    for i in range(B):
        w = webb[rng.integers(0, 6, G)][inv]
        ys = yd * w
        b = (xd @ ys) / xx
        e = ys - b * xd
        meat = np.bincount(inv, weights=xd * e, minlength=G)
        se = np.sqrt(adj * (meat ** 2).sum()) / xx
        tb[i] = b / se
    return float((np.abs(tb) >= abs(t_obs)).mean())


def hier_boot_slope(obs: dict, rng, B=N_BOOT) -> np.ndarray:
    """Resample repositories, then tasks within each repository; every
    slice observation of a resampled task comes along (keeps encoder
    dependence and Desc-source links). Percentiles are recomputed within
    each resample (see rerank)."""
    repo, task = obs["repo"], obs["task"]
    repos = np.unique(repo)
    idx_by_task = defaultdict(list)
    for i, t in enumerate(task):
        idx_by_task[(repo[i], t)].append(i)
    tasks_by_repo = defaultdict(list)
    for (r, t), ix in idx_by_task.items():
        tasks_by_repo[r].append(np.array(ix))
    out = np.empty(B)
    for b in range(B):
        pick = []
        for r in repos[rng.integers(0, len(repos), len(repos))]:
            tl = tasks_by_repo[r]
            for j in rng.integers(0, len(tl), len(tl)):
                pick.append(tl[j])
        ix = np.concatenate(pick)
        out[b] = fe_slope(obs["s"][ix], rerank(obs, ix), obs["y"][ix])
    return out


def task_permutation_p(obs: dict, names: list[str], rng, B=5000) -> dict:
    """Null distribution of the common slope that respects every
    dependence in the battery: within each (query pool, repository), tau
    percentiles are permuted across TASKS, and the same permutation is
    applied to every encoder slice of that pool (so the shared-query /
    shared-BM25 structure is intact; Desc travels as its own pool)."""
    s, x, y, repo, task = obs["s"], obs["x"], obs["y"], obs["repo"], obs["task"]
    pool = np.array(["_".join(names[i].split("_")[:2]) for i in s])
    key = {(s[i], task[i]): i for i in range(len(s))}
    groups = defaultdict(set)
    for i in range(len(s)):
        groups[(pool[i], repo[i])].add(task[i])
    groups = {k: sorted(v) for k, v in groups.items()}
    b0 = fe_slope(s, x, y)
    null = np.empty(B)
    for b in range(B):
        mp = {}
        for (p, r), ts in groups.items():
            perm = rng.permutation(len(ts))
            for a, j in zip(ts, perm):
                mp[(p, r, a)] = ts[j]
        xn = x.copy()
        for i in range(len(s)):
            j = key.get((s[i], mp[(pool[i], repo[i], task[i])]))
            if j is not None:
                xn[i] = x[j]
        null[b] = fe_slope(s, xn, y)
    p = (1 + (np.abs(null) >= abs(b0)).sum()) / (B + 1)
    return {"null_sd": float(null.std()), "p_two_sided": float(p), "B": B}


def slice_boot_slope(obs: dict, rng, B=N_BOOT) -> np.ndarray:
    """The paper's original slice-cluster bootstrap, for side-by-side."""
    k = obs["k"]
    per = [np.where(obs["s"] == g)[0] for g in range(k)]
    out = np.empty(B)
    for b in range(B):
        gs = rng.integers(0, k, k)
        ix = np.concatenate([per[g] for g in gs])
        sl = np.concatenate([np.full(len(per[g]), j) for j, g in enumerate(gs)])
        out[b] = fe_slope(sl, obs["x"][ix], obs["y"][ix])
    return out


def pooled_F_p(obs: dict) -> float:
    s, x, y = obs["s"], obs["x"], obs["y"]
    xd, yd = demean(s, x), demean(s, y)
    b = (xd @ yd) / (xd @ xd)
    sse0 = float(yd @ yd); sse1 = float(((yd - b * xd) ** 2).sum())
    n, k = len(y), obs["k"]
    F = (sse0 - sse1) / (sse1 / (n - k - 1))
    return float(stats.f.sf(F, 1, n - k - 1))


def summarize(bs: np.ndarray, est: float) -> dict:
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p_two = 2 * min((bs >= 0).mean(), (bs <= 0).mean())
    return {"est": est, "ci": [float(lo), float(hi)],
            "width": float(hi - lo), "p_boot_two_sided": float(max(p_two, 1 / len(bs)))}


def cross_cell_gap(slices: dict, rng, B=N_BOOT) -> dict:
    """Raw-Post minus Cur-Pre mean premium (jina-code-0.5b), query-level
    two-sample vs repository-block bootstrap."""
    a = slices["py_cur_jina05"]; b = slices["py_raw_jina05"]
    ya = np.array([r["delta"] for r in a]); yb = np.array([r["delta"] for r in b])
    ra = np.array([repo_of(r["query_id"]) for r in a])
    rb = np.array([repo_of(r["query_id"]) for r in b])
    gap = float(yb.mean() - ya.mean())
    q = np.empty(B)
    for i in range(B):
        q[i] = (yb[rng.integers(0, len(yb), len(yb))].mean()
                - ya[rng.integers(0, len(ya), len(ya))].mean())
    repos = np.unique(np.concatenate([ra, rb]))
    ia = {r: np.where(ra == r)[0] for r in repos}
    ib = {r: np.where(rb == r)[0] for r in repos}
    rbk = np.empty(B)
    for i in range(B):
        pick = repos[rng.integers(0, len(repos), len(repos))]
        xa = np.concatenate([ya[rng.choice(ia[r], len(ia[r]))] for r in pick if len(ia[r])])
        xb = np.concatenate([yb[rng.choice(ib[r], len(ib[r]))] for r in pick if len(ib[r])])
        rbk[i] = xb.mean() - xa.mean()
    # repo-balanced estimand: mean over repos of per-repo gap
    per_repo = {r: float(yb[ib[r]].mean() - ya[ia[r]].mean())
                for r in repos if len(ia[r]) and len(ib[r])}
    return {"gap": gap, "n_a": len(ya), "n_b": len(yb),
            "n_repos": len(repos),
            "query_level": summarize(q, gap),
            "repo_block": summarize(rbk, gap),
            "per_repo_gap": per_repo,
            "repo_balanced_gap": float(np.mean(list(per_repo.values())))}


CELLS = {"Cur-Pre": "canon202607/jina05_cellA_c1000",
         "Raw-Post": "canon202607/jina05_cellB_c1000",
         "Raw-Pre23": "r8_robust_jina-0.5b_c1000",
         "Raw-Pre22": "r8_2022_jina-0.5b_c1000"}


def cell_rows(rel: str) -> tuple[np.ndarray, np.ndarray]:
    ys, rs = [], []
    for f in sorted((SMOKE / rel).glob("*/smoke_test_django.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                ys.append(r["dense_ndcg10"] - r["bm25_ndcg10"])
                rs.append(repo_of(r["query_id"]))
    return np.array(ys), np.array(rs)


def contrast(hi: str, lo: str, rng, B=N_BOOT) -> dict:
    """Mean-premium contrast between two cells (jina-code-0.5b) with three
    intervals: query-level two-sample (the paper's), repository-block
    (resample repos, then queries within repo, both cells together), and a
    repository-stratified estimand (per-repo gap averaged with the
    lower cell's repo weights) under the same repository-block bootstrap."""
    yh, rh = cell_rows(CELLS[hi]); yl, rl = cell_rows(CELLS[lo])
    gap = float(yh.mean() - yl.mean())
    q = np.array([yh[rng.integers(0, len(yh), len(yh))].mean()
                  - yl[rng.integers(0, len(yl), len(yl))].mean() for _ in range(B)])
    repos = np.unique(np.concatenate([rh, rl]))
    ih = {r: np.where(rh == r)[0] for r in repos}
    il = {r: np.where(rl == r)[0] for r in repos}
    shared = [r for r in repos if len(ih[r]) and len(il[r])]
    w = np.array([len(il[r]) for r in shared], float); w /= w.sum()

    def strat(idx_h, idx_l):
        return float(sum(wi * (yh[idx_h[r]].mean() - yl[idx_l[r]].mean())
                         for wi, r in zip(w, shared)))
    st_hat = strat(ih, il)
    rb = np.empty(B); sb = np.empty(B)
    for b in range(B):
        pick = repos[rng.integers(0, len(repos), len(repos))]
        xh = [yh[rng.choice(ih[r], len(ih[r]))] for r in pick if len(ih[r])]
        xl = [yl[rng.choice(il[r], len(il[r]))] for r in pick if len(il[r])]
        rb[b] = np.concatenate(xh).mean() - np.concatenate(xl).mean()
        # stratified: repos resampled among shared, queries within repo
        pk = [shared[j] for j in rng.integers(0, len(shared), len(shared))]
        wk = np.array([len(il[r]) for r in pk], float); wk /= wk.sum()
        sb[b] = sum(wi * (yh[rng.choice(ih[r], len(ih[r]))].mean()
                          - yl[rng.choice(il[r], len(il[r]))].mean())
                    for wi, r in zip(wk, pk))
    return {"hi": hi, "lo": lo, "n_hi": int(len(yh)), "n_lo": int(len(yl)),
            "mean_hi": float(yh.mean()), "mean_lo": float(yl.mean()),
            "gap": gap, "n_repos": int(len(repos)), "n_shared_repos": len(shared),
            "query_level": summarize(q, gap),
            "repo_block": summarize(rb, gap),
            "repo_stratified": summarize(sb, st_hat)}


def main() -> None:
    rng = np.random.default_rng(0)
    slices = load_all()
    names = sorted(n for n in slices if n not in CHUNK_DUPES)
    print(f"loaded {len(slices)} slices, {len(names)} core")

    obs = within_repo_obs(slices, names)
    b_hat = fe_slope(obs["s"], obs["x"], obs["y"])
    print(f"n tau>0 obs (within-repo, core) = {len(obs['y'])}; "
          f"repos={len(np.unique(obs['repo']))}; "
          f"tasks={len(set(zip(obs['repo'], obs['task'])))}")
    print(f"common slope {b_hat:+.4f}   pooled-F p={pooled_F_p(obs):.2e}")

    # per-slice sign count (repro check)
    neg = 0
    for g, name in enumerate(names):
        m = obs["s"] == g
        if m.sum() >= 20 and np.polyfit(obs["x"][m], obs["y"][m], 1)[0] < 0:
            neg += 1
    print(f"negative per-slice slopes: {neg}/{len(names)}")

    old = summarize(slice_boot_slope(obs, rng), b_hat)
    new = summarize(hier_boot_slope(obs, rng), b_hat)
    b, se, t, G = cr1_t(obs["s"], obs["x"], obs["y"], obs["repo"])
    p_cr1 = float(2 * stats.t.sf(abs(t), G - 1))
    p_wcb = wild_cluster_p(obs["s"], obs["x"], obs["y"], obs["repo"], rng)
    print(f"slice-cluster CI  [{old['ci'][0]:+.4f},{old['ci'][1]:+.4f}]")
    print(f"repo>task CI      [{new['ci'][0]:+.4f},{new['ci'][1]:+.4f}]  "
          f"boot p={new['p_boot_two_sided']:.4g}")
    print(f"CR1 (repo, G={G}) se={se:.4f} t={t:.2f} p={p_cr1:.2e}; "
          f"wild-cluster p={p_wcb:.4g}")
    perm = task_permutation_p(obs, names, rng)
    print(f"task-block permutation: null sd={perm['null_sd']:.4f} "
          f"p={perm['p_two_sided']:.4g} (B={perm['B']})")

    # ---- leave-one-repo-out sensitivity on the slope ----
    loro = {}
    for r in np.unique(obs["repo"]):
        m = obs["repo"] != r
        loro[r] = fe_slope(obs["s"][m], obs["x"][m], obs["y"][m])
    print("leave-one-repo-out slope range "
          f"[{min(loro.values()):+.4f}, {max(loro.values()):+.4f}]")

    # ---- #2: slices untouched by predictor selection ----
    # MaxIDF was chosen on Cur-Pre/Raw-Post (tab:r3, jina05 + the R4 encoder
    # fan-out on the same queries) and, per the paper, partly on transport
    # across the 38-slice battery. Slices whose QUERIES never entered either
    # step before the choice: none can be certified for the battery; the
    # closest to untouched are the query pools other than Cur-Pre/Raw-Post.
    holdout_pools = [n for n in names
                     if not (n.startswith("py_cur_") or n.startswith("py_raw_"))]
    h = within_repo_obs(slices, holdout_pools)
    hb = fe_slope(h["s"], h["x"], h["y"])
    hs = summarize(hier_boot_slope(h, rng), hb)
    hb2, hse, ht, hG = cr1_t(h["s"], h["x"], h["y"], h["repo"])
    print(f"non-A/B pools ({len(holdout_pools)} slices, n={len(h['y'])}, "
          f"repos={hG}): slope {hb:+.4f} CI [{hs['ci'][0]:+.4f},{hs['ci'][1]:+.4f}] "
          f"CR1 p={2*stats.t.sf(abs(ht), hG-1):.3g}")
    per_pool = {}
    for n in holdout_pools:
        o = within_repo_obs(slices, [n])
        if len(o["y"]) < 20:
            continue
        bb = fe_slope(o["s"], o["x"], o["y"])
        ss = summarize(hier_boot_slope(o, rng, B=2000), bb)
        per_pool[n] = {"n": int(len(o["y"])),
                       "repos": int(len(np.unique(o["repo"]))), **ss}
        print(f"   {n:18s} n={len(o['y']):4d} slope {bb:+.3f} "
              f"CI [{ss['ci'][0]:+.3f},{ss['ci'][1]:+.3f}]")

    contrasts = {}
    for hi, lo in (("Raw-Post", "Cur-Pre"), ("Raw-Pre23", "Cur-Pre"),
                   ("Raw-Pre22", "Cur-Pre"), ("Raw-Post", "Raw-Pre23"),
                   ("Raw-Post", "Raw-Pre22")):
        c = contrast(hi, lo, rng)
        contrasts[f"{hi} - {lo}"] = c
        f = lambda d: f"[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]"
        print(f"{hi:9s} - {lo:9s} n={c['n_hi']}/{c['n_lo']} "
              f"means {c['mean_hi']:+.3f}/{c['mean_lo']:+.3f} gap {c['gap']:+.4f} "
              f"query {f(c['query_level'])} repo-block {f(c['repo_block'])} "
              f"| stratified {c['repo_stratified']['est']:+.4f} {f(c['repo_stratified'])}")
    gap = cross_cell_gap(slices, rng)
    print(f"Raw-Post minus Cur-Pre (jina05) gap {gap['gap']:+.4f}: "
          f"query-level CI {gap['query_level']['ci']}, "
          f"repo-block CI {gap['repo_block']['ci']}; "
          f"repo-balanced gap {gap['repo_balanced_gap']:+.4f}")

    out = {
        "variant": VARIANT, "core_slices": names,
        "n_obs": int(len(obs["y"])),
        "n_repos": int(len(np.unique(obs["repo"]))),
        "common_slope": b_hat,
        "pooled_F_p": pooled_F_p(obs),
        "slice_cluster_bootstrap": old,
        "repo_task_bootstrap": new,
        "cr1_repo": {"se": se, "t": t, "G": G, "p": p_cr1},
        "wild_cluster_bootstrap_p": p_wcb,
        "task_block_permutation": perm,
        "note_cr1": ("CR1-by-repo SE (~0.010) is anti-conservative here: "
                     "19 clusters, django = 38% of rows, and rank-based x "
                     "makes task scores negatively correlated within repo. "
                     "Task-level bootstrap and task-block permutation agree "
                     "on SE ~0.04; use those."),
        "leave_one_repo_out": loro,
        "non_AB_pools": {"slices": holdout_pools, "n": int(len(h["y"])),
                         "slope": hb, **hs,
                         "cr1_p": float(2 * stats.t.sf(abs(ht), hG - 1)),
                         "per_pool": per_pool},
        "cross_cell_gap_jina05_tau_joined": gap,
        "cell_contrasts_jina05": contrasts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
