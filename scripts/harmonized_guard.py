"""Guard: the harmonized runs must reproduce the canonical top-10 lists for every
query that was already in the original pool (rankings do not depend on gold)."""
import json, sys
from pathlib import Path


def load(d, field):
    out = {}
    for f in Path(d).glob("*/smoke_test_django.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if field in r:
                out[r["query_id"]] = r[field]
    return out


def main():
    canon, new, field = sys.argv[1], sys.argv[2], sys.argv[3]
    c, n = load(canon, field), load(new, field)
    shared = sorted(set(c) & set(n))
    same = sum(c[q] == n[q] for q in shared)
    print(f"{field}: canon {len(c)} new {len(n)} shared {len(shared)} identical {same}"
          f" missing_from_new {len(set(c) - set(n))}")
    for q in [q for q in shared if c[q] != n[q]][:5]:
        print("  DIFF", q, c[q][:3], n[q][:3])


if __name__ == "__main__":
    main()
