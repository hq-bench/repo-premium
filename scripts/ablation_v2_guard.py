"""Guard for the fixed anchor ablation: unchanged-anchor queries must be byte-identical to the old run."""
from paths import DATA_ROOT
import glob, json, sys, os
D = strDATA_ROOT
OLD = {"B": f"{D}/ablation/qrels_cellB_ablated", "rust": f"{D}/ablation_r9/rust", "ts": f"{D}/ablation_r9/ts"}
ok = True
for sl, od in OLD.items():
    n = same_anchor = ident = ident_ties = changed = was_noop = now_noop = 0
    bad = []
    for f in sorted(glob.glob(f"qrels_{sl}/*_audit.jsonl")):
        repo = os.path.basename(f)[:-len("_audit.jsonl")]
        aud = {json.loads(l)["query_id"]: json.loads(l) for l in open(f)}
        new = {json.loads(l)["query_id"]: json.loads(l) for l in open(f"qrels_{sl}/{repo}.jsonl")}
        old = {json.loads(l)["query_id"]: json.loads(l) for l in open(f"{od}/{repo}.jsonl")}
        olda = {json.loads(l)["query_id"]: json.loads(l) for l in open(f"{od}/{repo}_audit.jsonl")}
        for q, r in new.items():
            n += 1
            same = r["gold_files"][0] == aud[q]["anchor_gold_file"]
            idt = r["ablated_query"] == old[q]["ablated_query"] and r["control_query"] == old[q]["control_query"]
            # tie-insensitive: same ablated tokens, same placebo IDF profile
            na_, oa_ = aud[q], olda[q]
            prof = lambda a, k: sorted(x[1] for x in a[k])
            idt_ties = (prof(na_, "ablated_tokens") == prof(oa_, "ablated_tokens")
                        and prof(na_, "control_tokens") == prof(oa_, "control_tokens"))
            was_noop += olda[q]["n_shared_before"] == 0
            now_noop += aud[q]["n_shared_before"] == 0
            if same:
                same_anchor += 1; ident += idt; ident_ties += idt_ties
                if not idt_ties: bad.append(q)
            else:
                changed += not idt
    print(f"{sl:5s} n={n} same-anchor={same_anchor} byte-identical={ident} tie-equivalent={ident_ties} | new-anchor={n-same_anchor} "
          f"(text changed {changed}) | empty-gold before {was_noop} -> after {now_noop}")
    if bad:
        print("   NOT IDENTICAL:", bad[:5]); ok = False
print("GUARD", "OK" if ok else "FAILED"); sys.exit(0 if ok else 1)
