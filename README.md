# RepoPremium

RepoPremium is the benchmark and experiment code for **“Embeddings vs. Grep: Rethinking the Dense Advantage in Code Search”**. It compares lexical and dense code retrieval across query construction methods, repository snapshots, languages, and agent workflows, with cached-result analyses for the paper’s tables and figures.

## Install

Use Python 3.10+ and Git. Some corpus-mining tools also require the GitHub CLI (`gh`) and authenticated access; cached analyses do not. GPU retrieval requires a suitable PyTorch installation; the requirements do not request CUDA extras.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

**The dataset is not yet public.** The planned download command is:

```bash
huggingface-cli download hq-bench/repo-premium --repo-type dataset --local-dir data
```

The final archive layout and availability remain TODO. Model weights are downloaded separately by the retrieval libraries. By default, `scripts/paths.py` reads from this repository’s `data/`, independent of the current working directory. To use an external data directory:

```bash
export REPO_PREMIUM_DATA="/path/to/repo-premium-data"
```

Data, repository checkouts, and model caches are excluded from this code release. Preserve the flat `scripts/` layout and its `scaleup/`, `r9_multilang/`, `deepswe/`, and `analysis_helpers/` subdirectories: scripts import sibling modules.

## Run retrieval

Run from the repository root with a qrel file and the matching repository snapshot:

```bash
python scripts/run_retrieval.py \
  --qrel-file "$REPO_PREMIUM_DATA/release/qrels/django__django.jsonl" \
  --clone-dir "$REPO_PREMIUM_DATA/repos/django__django" \
  --out-dir results/django \
  --dense-model jinaai/jina-code-embeddings-0.5b \
  --chunk-chars 1000 --n 1000
```

This evaluates grep, BM25, and dense retrieval. PR boilerplate stripping is **on by default**; use `--no-strip-boilerplate` to disable it or `--skip-dense` for lexical retrieval only. `--code-exts` and `--tokenize-camelcase` support other languages. See `--help` for model, chunking, and query options. The three-line `smoke_test_django.py` entry point preserves legacy imports and invocations; cached output filenames retain `smoke_test_django.jsonl` for compatibility.

[TABLES.md](TABLES.md) maps paper labels to regeneration commands and identifies unverified mappings and missing producers. Some analyses write under the data root: use a writable copy for reproduction, or a script’s `--out` option to redirect results.

## Agent-evaluation harness

`agent_clean_env.py` materializes a clean task environment from the base commit, removes access to repository history that could reveal the fix, installs PATH shims to enforce search restrictions, and audits trajectories for search bypasses. These controls support the experiment protocol; they do not provide operating-system isolation.

`run_agent_eval.py` runs the direct-shell arm; `run_agent_eval_wrapper.py` runs the `grep`, `dense`, and `router` wrapper arms. Both invoke `mini-swe-agent` from PATH, overridable with `MINI_SWE_AGENT_BIN`. Set the model provider credentials in the environment and select a provider/model supported by your installation with `--model`. `AGENT_WORKTREE_ROOT` overrides the task workspace root, which otherwise lives under the data root.

`coreb_search.py` builds and queries the wrapper’s search indices:

```bash
python scripts/coreb_search.py build \
  --repo "$REPO_PREMIUM_DATA/repos/django__django" \
  --index-dir results/indices/django__django --mode both
python scripts/coreb_search.py query \
  --index-dir results/indices/django__django --mode dense \
  --query 'database cache deletion' --k 20
python scripts/run_agent_eval.py --help
python scripts/run_agent_eval_wrapper.py --help
```

For direct wrapper retrieval, set `COREB_SEARCH_PY` to an interpreter with the retrieval dependencies installed. Otherwise configure `APPTAINER_IMAGE` and `APPTAINER_BIND` for Apptainer; the bind defaults to the data root and code checkout mounted at the same paths. Include any index directories outside those roots in the bind list. `COREB_SEARCH_SCRIPT` overrides the default sibling `coreb_search.py` path. Container images and server launch scripts are not distributed.

Generative query and curation tools use an OpenAI-compatible endpoint supplied through `OPENAI_API_BASE` and `OPENAI_API_KEY`. Set the full API base URL (including `/v1` if required by the endpoint). There is no default endpoint; keep credentials in the environment.

## Citation

TODO: add the paper’s bibliographic citation and public identifier.

## License

Code: [Apache-2.0](LICENSE). Benchmark data: CC-BY-4.0, on top of the source repositories’ licenses; source code retains its original license obligations.
