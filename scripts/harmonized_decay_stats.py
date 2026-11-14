"""fig:decay statistics (pooled Cur-Pre + Raw-Post, tau_top1 > 0): paper inputs vs harmonized Raw-Post with fixed tau."""
import glob, json
import numpy as np
def recs(d):
    return {json.loads(l)["query_id"]: json.loads(l) for f in glob.glob(d + "/*/smoke_test_django.jsonl") for l in open(f)}
def taus(d):
    return {json.loads(l)["query_id"]: json.loads(l)["tau_top1"] for f in glob.glob(d + "/*/tau_variants.jsonl") for l in open(f)}
def stats(pairs, name):
    t = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs]); m = t > 0; t, y = t[m], y[m]
    q = np.quantile(t, [.25, .75])
    print(f"{name}: n(tau>0)={m.sum()} r={np.corrcoef(t, y)[0,1]:+.3f} Q1={y[t <= q[0]].mean():+.3f} Q4={y[t > q[1]].mean():+.3f} "
          f"slope={np.polyfit(t, y, 1)[0]:+.4f}")
A, B = recs("_paper_indexed/canon/jina05_cellA_c1000"), recs("_paper_indexed/canon/jina05_cellB_c1000")
tA, tB = taus("_paper_indexed/tau_all/tau_variants_canon202607/cellA"), taus("_paper_indexed/tau_all/tau_variants_canon202607/cellB")
pa = [(tA[q], r["dense_ndcg10"] - r["bm25_ndcg10"]) for q, r in A.items() if q in tA]
pb = [(tB[q], r["dense_ndcg10"] - r["bm25_ndcg10"]) for q, r in B.items() if q in tB]
stats(pa + pb, "paper (A + B, paper tau)")
stats(pa + [(t, d) for t, d in pb], "  check")
rows = [json.loads(l) for l in open("harmonized_rows.jsonl")]
tNB = taus("out/tau_cellB")
hb = [(tNB[r["qid"]], r["prem_c1000"]) for r in rows if r["pool"] == "Raw-Post"]
hb_old = [(tNB[r["qid"]], r["prem_c1000"]) for r in rows if r["pool"] == "Raw-Post" and r["old"]]
stats(pa + hb_old, "paper B queries, fixed tau")
stats(pa + hb, "harmonized B, fixed tau")
