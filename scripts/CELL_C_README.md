# Cell C: descriptive query controls

`build_cell_c.py` rewrites symbol-named queries into descriptive queries;
`build_cell_c_qrels.py` emits per-repository qrels. Use an OpenAI-compatible
endpoint via `OPENAI_API_BASE` (including `/v1` if required) and `OPENAI_API_KEY`.
Supply credentials through the environment.

```bash
python scripts/build_cell_c.py --help
python scripts/build_cell_c_qrels.py --help
python scripts/run_retrieval.py --help
python scripts/aggregate_cell_c.py --help
```

Run retrieval once per repository and encoder with the generated qrels, then
aggregate cached results with `aggregate_cell_c.py`. Its `--results-root`,
`--graph-root`, `--rerank-root`, `--strat-pooled`, `--cell-c`, and `--out`
flags select the inputs and output. Data defaults are under `REPO_PREMIUM_DATA`.
The original server launchers are excluded from this release.
