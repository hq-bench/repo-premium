"""One probe qrel file per (slice, repo): each query twice, as <id>|ablated and <id>|control, in
field probe_query, so one dense pass embeds the corpus once for both arms. Writes jobs.tsv."""
from paths import DATA_ROOT
import glob, json, os
W = str(DATA_ROOT / 'ablation_v2'); D = strDATA_ROOT
CFG = {"B": (".py", ""), "rust": (".rs", ""), "ts": (".ts,.tsx,.js,.mjs,.cjs", "--tokenize-camelcase --max-seq-length 1024")}
jobs = []
for sl, (exts, extra) in CFG.items():
    os.makedirs(f"{W}/probe_{sl}", exist_ok=True)
    for f in sorted(glob.glob(f"{W}/qrels_{sl}/*.jsonl")):
        if f.endswith("_audit.jsonl"):
            continue
        repo = os.path.basename(f)[:-6]
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if not rows:
            continue
        out = []
        for r in rows:
            for arm in ("ablated", "control"):
                out.append(json.dumps({**r, "query_id": f"{r['query_id']}|{arm}", "probe_query": r[f"{arm}_query"]}))
        open(f"{W}/probe_{sl}/{repo}.jsonl", "w").write("\n".join(out) + "\n")
        jobs.append("\t".join([sl, repo, f"{W}/probe_{sl}/{repo}.jsonl", f"{D}/repos/{repo}", exts, extra]))
# TypeScript first: it is by far the longest task.
jobs.sort(key=lambda j: j.split("\t")[0] != "ts")
open(f"{W}/jobs.tsv", "w").write("\n".join(jobs) + "\n")
print(len(jobs), "tasks"); print("\n".join(j.split("\t")[0] + " " + j.split("\t")[1] for j in jobs))
