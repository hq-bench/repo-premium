"""tab:budget dev column: Cur-Pre (paper lists) + Raw-Post (paper 132 vs harmonized 288). Repo-block bootstrap, A/B django share a fold."""
import glob, json, math, random
def load(d):
    return {json.loads(l)["query_id"]: dict(json.loads(l), _repo=f.split("/")[-2]) for f in glob.glob(d + "/*/smoke_test_django.jsonl") for l in open(f)}
def ndcg(t, g):
    g = set(g); return sum(1/math.log2(i+2) for i, f in enumerate(t[:10]) if f in g) / sum(1/math.log2(i+2) for i in range(min(len(g), 10)))
def rrf(a, b, k=60):
    s, first = {}, {}
    for li, l in enumerate((a, b)):
        for r, x in enumerate(l[:10]):
            s[x] = s.get(x, 0) + 1/(k+r+1); first.setdefault(x, (r, li))
    return sorted(s, key=lambda x: (-s[x], first[x]))[:10]
def rows_for(dense, bm, gen, gold_src):
    out = []
    for q in dense:
        if q not in bm or q not in gen: continue
        g = gold_src[q]; d, b, e = dense[q]["dense_top10"], bm[q]["bm25_top10"], gen[q]["bm25_top10"]
        out.append({"repo": dense[q]["_repo"], "dense": ndcg(d, g), "bm25": ndcg(b, g), "rrf_db": ndcg(rrf(d, b), g),
                    "gen_t": ndcg(e, g), "rrf_dgt": ndcg(rrf(d, e), g)})
    return out
def report(rows, name):
    rng = random.Random(20261010); grp = {}
    for r in rows: grp.setdefault(r["repo"], []).append(r)
    keys = sorted(grp); res = {}
    for k in ["bm25", "dense", "rrf_db", "gen_t", "rrf_dgt"]:
        m = sum(r[k] for r in rows) / len(rows); dd = []
        for _ in range(10000):
            s = [r for _ in keys for r in grp[keys[rng.randrange(len(keys))]]]
            dd.append(sum(r[k] - r["dense"] for r in s) / len(s))
        dd.sort(); res[k] = (m, m - sum(r["dense"] for r in rows)/len(rows), dd[250], dd[9750])
    print(f"{name}: n={len(rows)} repos={len(keys)}")
    for k, (m, md, lo, hi) in res.items(): print(f"  {k:8s} {m:.3f}  minus_dense {md:+.3f} [{lo:+.3f},{hi:+.3f}]")
P = "_paper_indexed/"
A = load(P + "canon/jina05_cellA_c1000")
rA = rows_for(A, load(P + "bm25_sweep202610/bm25_cellA_c1000000000"), load(P + "gen_sweep202610/gen_cellA_c1000000000"), {q: r["gold_files"] for q, r in A.items()})
B = load(P + "canon/jina05_cellB_c1000")
rB = rows_for(B, load(P + "bm25_sweep202610/bm25_cellB_c1000000000"), load(P + "gen_sweep202610/gen_cellB_c1000000000"), {q: r["gold_files"] for q, r in B.items()})
report(rA + rB, "paper dev (A + B132)")
H = {json.loads(l)["query_id"]: json.loads(l)["gold_files"] for f in glob.glob("qrels_harmonized/cellB/*.jsonl") for l in open(f)}
dH = {q: r for q, r in load("out/dense_all_c1000").items() if q in H}
rH = rows_for(dH, load("out/bm25_cellB_c1000000000"), load("out/gen_cellB_c1000000000"), H)
report(rA + rH, "harmonized dev (A + B288)")
