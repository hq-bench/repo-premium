# Multi-language replication

The cached raw pre-cutoff replication uses microsoft/TypeScript (TS/JS),
gohugoio/hugo (Go), and tokio-rs/tokio (Rust). Generate qrels with
`extract_qrels_from_post_cutoff_prs.py` and run `run_retrieval.py` for each
repository with the appropriate `--code-exts`. TS/JS uses
`--tokenize-camelcase`.

```bash
export REPO_PREMIUM_DATA="${REPO_PREMIUM_DATA:-$PWD/data}"
python scripts/r9_multilang/aggregate_multilang.py
```

Aggregation reads `smoke_test_results/r9b_ts_c1000`, `r9c_go_c1000`, and
`r9d_rust_c1000` beneath the data root. It reports per-language quartile
means, pooled dense premiums, bootstrap intervals, and monotonicity checks.
