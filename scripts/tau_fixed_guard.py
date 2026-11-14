"""Guard for the fixed-tau recompute: per pool, paper tau vs fixed tau.

A query whose paper tau came from a usable gold_files[0] (indexed, not a test) must reproduce
its paper tau exactly; this checks each pool's snapshot, extensions and tokenizer. Every other
change must be explained by gold_files[0] being unindexed (paper tau had an empty gold set)
or a test file. Run from the work dir after run_tau_fixed.sh; exits 1 on any unexplained
change or missing query.
"""
from paths import DATA_ROOT
import glob
import json
import sys
from pathlib import Path

from build_source_only_mirror import is_test

D = DATA_ROOT
P = Path("_paper_indexed/tau_all")
POOLS = {"cellA": ("tau_variants_canon202607/cellA", "qrels"),
         "cellB": ("tau_variants_canon202607/cellB", "qrels_post_cutoff"),
         "r5": ("tau_variants_law/r5", "qrels_r5"),
         "r8_2022": ("tau_variants_law/r8_2022", "qrels_r8_2022"),
         "heldout": ("tau_variants_law/heldout", "qrels_heldout"),
         "cellC": ("tau_variants_law/cellC", "qrels_cellC"),
         "r9_vuejs": ("tau_variants_law/r9_vuejs", "qrels_r9_vuejs_2023"),
         "r9b_ts": ("tau_variants_law/r9b_ts", "qrels_r9b_ts"),
         "r9c_go": ("tau_variants_law/r9c_go", "qrels_r9c_go"),
         "r9d_rust": ("tau_variants_law/r9d_rust", "qrels_r9d_rust")}


def taus(d) -> dict:
    return {json.loads(l)["query_id"]: json.loads(l)
            for f in glob.glob(f"{d}/*/tau_variants.jsonl") for l in open(f) if l.strip()}


def gold0(qdir: Path) -> dict:
    out = {}
    for f in qdir.glob("*.jsonl"):
        if f.name.startswith("."):
            continue
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("gold_files"):
                    out[r["query_id"]] = r["gold_files"][0]
    return out


def main() -> int:
    ok = True
    for pool, (pdir, qdir) in POOLS.items():
        old, new, g0 = taus(P / pdir), taus(f"tau_fixed/{pool}"), gold0(D / qdir)
        missing = sorted(set(old) - set(new))
        changed = [q for q in old if q in new and abs(old[q]["tau_top1"] - new[q]["tau_top1"]) > 1e-9]
        empty = [q for q in changed if old[q]["n_gold_tokens"] == 0]
        test = [q for q in changed if q not in empty and is_test(g0.get(q, ""))]
        bad = [q for q in changed if q not in empty and q not in test]
        z_old = sum(old[q]["tau_top1"] == 0 for q in old)
        z_new = sum(new[q]["tau_top1"] == 0 for q in new)
        print(f"{pool:9s} n {len(old):4d}  identical {len(old) - len(changed) - len(missing):4d}  "
              f"changed {len(changed):3d} = empty-gold {len(empty):3d} + test-first {len(test):3d} "
              f"+ UNEXPLAINED {len(bad)}  | tau=0 {z_old:3d} -> {z_new:3d}")
        if missing:
            print("   MISSING:", missing[:6])
        if bad:
            print("   unexplained:", [(q, g0.get(q), round(old[q]["tau_top1"], 3), round(new[q]["tau_top1"], 3))
                                     for q in bad[:5]])
        ok = ok and not missing and not bad
    print("GUARD", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
