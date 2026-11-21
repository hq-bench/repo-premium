# Paper tables and figures

Maps each paper table/figure label to the scripts and cached inputs that regenerate it.
Entries marked TODO are not yet tied to a producer; the mapping is being verified
against the final paper numbers.

These commands require the original experiment layout. Download dataset revision
`97c0c94062634d8266a7238546169123a8ef2125` (tag `paper-artifacts`), as shown in
[README.md](README.md#download-the-paper-artifacts). The dataset’s new default
Parquet layout is not a drop-in replacement for these cached inputs.

Run from the repository root:

```bash
export RELEASE_ROOT="$PWD"
export REPO_PREMIUM_DATA="${REPO_PREMIUM_DATA:-$PWD/data}"
export OUT="$PWD/results/tables"
export HARMONIZED_WORK="$REPO_PREMIUM_DATA/smoke_test_results/harmonized202610"
mkdir -p "$OUT"
```

Use an absolute data path. Some legacy analyses write JSON under `analysis/`
or into their working directory, and plots write under `figures/coreb_repo/`.
Use a writable data copy for these commands. Commands with `--out` can read
read-only caches. Optional retrieval rebuilds need corpus checkouts and model
weights; generative/agent reruns also need model credentials.

Uppercase variables not set above (`CLONE`, `QREL`, `TASKS`, `TASK_OUT`,
`TAU_OUT`, `SAMPLE`, `INDICES`, `INDEX`, `RUNS`, `AGENT_MODEL`,
`LOCKBOX_MANIFEST`, `QPP_OUT`, `SCORES`, `RERANK_OUT`, `SCALEUP_TSV`) are explicit input/output
placeholders: set them to the matching released artifact before execution.
TODO: verify the public dataset names and complete batch invocations. The
harmonized work directory must contain `_paper_indexed/`, `out/`,
`qrels_harmonized/`, and intermediate harmonized rows as needed by each script.

## fig:overview

Producers: plot_fig_premium_overview.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: analysis/harmonized_overview.json, analysis/indexed/genarm_rebase.json.

```bash
python scripts/plot_fig_premium_overview.py
```

TODO: verify the producer assembling `analysis/harmonized_overview.json`; the plot consumes that cached summary.

## fig:example-queries

Producers: hand-written LaTeX.

Cached inputs beneath `REPO_PREMIUM_DATA`: -.

```bash
# Hand-written LaTeX; no automated producer. TODO: release the LaTeX source.
```

## fig:construction

Producers: plot_fig_construction.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: none.

```bash
python scripts/plot_fig_construction.py
```

## tab:cells

Producers: mediation_decomposition.py, chunk_rebase.py; file counts from local_inputs_202610/cached_nfiles.tsv.

Cached inputs beneath `REPO_PREMIUM_DATA`: smoke_test_results/canon202607, tau_variants_c1000.

```bash
python scripts/mediation_decomposition.py --pull "$REPO_PREMIUM_DATA/smoke_test_results" --cell-a canon202607/jina05_cellA_c1000 --cell-b canon202607/jina05_cellB_c1000 --tau tau_variants_c1000 --out "$OUT/mediation.json"
python scripts/chunk_rebase.py --out "$OUT/chunk_rebase.json"
cat "$REPO_PREMIUM_DATA/local_inputs_202610/cached_nfiles.tsv"
```

## tab:genarm-tuned

Producers: genarm_rebase.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: analysis/genarm_rebase.json, remote_pull/genarm/results.

```bash
python scripts/genarm_rebase.py --out "$OUT/genarm_rebase.json"
```

## tab:budget

Producers: harmonized_dev_budget.py, measure_index_cost.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: smoke_test_results/harmonized202610, canon202607.

```bash
(cd "$HARMONIZED_WORK" && python "$RELEASE_ROOT/scripts/harmonized_dev_budget.py")
python scripts/measure_index_cost.py --clone-dir "$REPO_PREMIUM_DATA/repos/django__django"
```

## fig:decay

Producers: plot_fig_intro_regimes.py; stats harmonized_decay_stats.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: tau_variants_c1000, harmonized202610.

```bash
python scripts/plot_fig_intro_regimes.py
(cd "$HARMONIZED_WORK" && python "$RELEASE_ROOT/scripts/harmonized_decay_stats.py")
```

The plotting script expects legacy `cellA/`, `cellB/`, and `tau_variants/` caches under `smoke_test_results/`; TODO: verify their relationship to the canonical c1000 tree before using the figure in the paper.

## tab:r8

Producers: plot_fig9_r8_decomposition.py, harmonized_extras.py, aggregate_r5.py, analysis_helpers/aggregate_r8_2022.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: smoke_test_results/q2_*, r8_*, r5_*.

```bash
python scripts/plot_fig9_r8_decomposition.py
(cd "$HARMONIZED_WORK" && python "$RELEASE_ROOT/scripts/harmonized_extras.py")
python scripts/aggregate_r5.py --r5-root "$REPO_PREMIUM_DATA/smoke_test_results" --out "$OUT/r5.json"
python scripts/analysis_helpers/aggregate_r8_2022.py
```

## tab:funnel, tab:repostats

Producers: UNKNOWN.

Cached inputs beneath `REPO_PREMIUM_DATA`: release/.

```bash
# TODO: producer UNKNOWN; inputs are under release/.
```

## tab:percommit

Producers: percommit_retrieval.py -> percommit_analyze.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: percommit202610/out, analysis/indexed/percommit_premium.json.

```bash
# To rebuild retrieval: run per repository/pool; TASKS and CLONE must match.
python scripts/percommit_retrieval.py --repo-dir "$CLONE" --tasks "$TASKS" --out "$TASK_OUT"
python scripts/percommit_analyze.py
```

## tab:test-gold

Producers: test_gold_audit.py, test_gold_symmetric.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: analysis/test_gold_audit.json, test_gold_symmetric.json.

```bash
python scripts/test_gold_audit.py
python scripts/test_gold_symmetric.py
```

## tab:anchor-split

Producers: anchor_type_split.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: smoke_test_results/anchor_split202610, local_inputs_202610/anchor_split_qrels.

```bash
python scripts/anchor_type_split.py analyze
# To rebuild controls, run anchor_type_split.py build, then run_retrieval.py
# for each emitted qrel with --skip-dense. TODO: verify batch naming.
```

## tab:encoder2

Producers: encoder2_splade.py, splade_eval.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: analysis/encoder2_splade.json, smoke_test_results/splade_202610.

```bash
python scripts/encoder2_splade.py
# Optional retrieval rebuild (requires model weights and corpus checkouts):
python scripts/splade_eval.py --data "$REPO_PREMIUM_DATA" --out "$REPO_PREMIUM_DATA/smoke_test_results/splade_202610"
```

## tab:metrics

Producers: metrics_robustness.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: analysis/metrics_robustness.json.

```bash
python scripts/metrics_robustness.py
```

## tab:gen-controls

Producers: rm3_bm25.py, rm3_compare.py, hyde_compare.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: smoke_test_results/rm3_202610, hyde_202610.

```bash
python scripts/rm3_compare.py
python scripts/hyde_compare.py
# Optional RM3 rebuild:
python scripts/rm3_bm25.py --qrels-root "$REPO_PREMIUM_DATA/local_inputs_202610" --repos "$REPO_PREMIUM_DATA/local_inputs_202610/repos_py" --repos-2024q1 "$REPO_PREMIUM_DATA/local_inputs_202610/repos_2024q1_py" --out "$REPO_PREMIUM_DATA/smoke_test_results/rm3_202610"
```

## tab:r3

Producers: recompute_tau_variants.py, law_clustered_bootstrap.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: smoke_test_results/tau_variants_*.

```bash
python scripts/law_clustered_bootstrap.py
# Optional tau rebuild per repository/cell:
python scripts/recompute_tau_variants.py --qrel-file "$QREL" --clone-dir "$CLONE" --out-dir "$TAU_OUT" --strip-boilerplate --chunk-chars 1000
```

## tab:r5, tab:chunk-rebase

Producers: chunk_rebase.py, chunk_rebase_ext.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: smoke_test_results/cellA_chunk_sweep, cellA_chunked, cellB_chunked.

```bash
python scripts/chunk_rebase.py --out "$OUT/chunk_rebase.json"
python scripts/chunk_rebase_ext.py --out "$OUT/chunk_rebase_ext.json"
```

## tab:curated-multiling

Producers: r9_multilang/aggregate_multilang.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: smoke_test_results/r9b_ts_c1000, r9c_go_c1000, r9d_rust_c1000, cur_*.

```bash
python scripts/r9_multilang/aggregate_multilang.py
```

The named aggregator currently covers raw TS/Go/Rust runs only. TODO: identify the producer for the curated `cur_*` rows.

## tab:perrepo

Producers: UNKNOWN.

Cached inputs beneath `REPO_PREMIUM_DATA`: smoke_test_results/canon202607.

```bash
# TODO: producer UNKNOWN; inputs are smoke_test_results/canon202607/.
```

## fig:law

Producers: plot_fig_law.py, law_clustered_bootstrap.py, law_confirmatory.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: analysis/law_*.json, smoke_test_results/tau_variants_law.

```bash
python scripts/law_clustered_bootstrap.py
python scripts/law_confirmatory.py
python scripts/plot_fig_law.py
```

## tab:scaleup

The cached aggregation command below is for jina-0.5b; repeat for qwen3-0.6b
with a separate output. For optional tau recomputation, `SCALEUP_TSV` must be
the original corpus-bearing `eligible_main.tsv`, not the JSONL split manifest.
TODO: release or reconstruct that TSV with portable corpus paths.

Producers: scaleup/analyze_scaleup.py, scaleup/tau_top1_scaleup.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: scaleup_pull/.

```bash
python scripts/scaleup/analyze_scaleup.py --retrieval-root "$REPO_PREMIUM_DATA/scaleup_pull/experiment_v1/retrieval/jina-0.5b" --manifest "$REPO_PREMIUM_DATA/scaleup_pull/splits_v2/retrieval_tasks_v2.jsonl" --out "$OUT/scaleup.json"
# Optional tau rebuild (requires matching corpus snapshots):
python scripts/scaleup/tau_top1_scaleup.py --manifest "$SCALEUP_TSV" --smoke "$RELEASE_ROOT/scripts" --out "$OUT/tau_top1_scaleup.jsonl"
```

## tab:router-pred

Producers: decision_model_v2.py, decision_model_v2_lockbox.py, qpp_scores.py, qpp_lockbox.py, router_ceiling.py, router_llm_judge.py, rerank_union.py, gen_tier_lockbox.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: analysis/decision_model_v2_*, smoke_test_results/qpp_*, gen_sweep202610.

```bash
python scripts/decision_model_v2.py headroom
python scripts/decision_model_v2.py route --clones "$REPO_PREMIUM_DATA/repos"
python scripts/decision_model_v2.py qpp
python scripts/decision_model_v2.py freeze
python scripts/decision_model_v2_lockbox.py guard
python scripts/decision_model_v2_lockbox.py score
python scripts/gen_tier_lockbox.py guard
python scripts/gen_tier_lockbox.py score
python scripts/router_ceiling.py --clones "$REPO_PREMIUM_DATA/repos" --out "$OUT/router_ceiling.json"
python scripts/router_llm_judge.py analyze --device cpu --out "$OUT/router_llm_judge.json"
# Optional feature/reranking cache rebuilds:
python scripts/qpp_scores.py --clones "$REPO_PREMIUM_DATA/repos" --qrels-a "$REPO_PREMIUM_DATA/local_inputs_202610/qrels" --qrels-b "$REPO_PREMIUM_DATA/local_inputs_202610/qrels_post_cutoff"
python scripts/qpp_lockbox.py --manifest "$LOCKBOX_MANIFEST" --out "$QPP_OUT"
python scripts/rerank_union.py --scores "$SCORES" --clones "$REPO_PREMIUM_DATA/repos" --qrels-a "$REPO_PREMIUM_DATA/local_inputs_202610/qrels" --qrels-b "$REPO_PREMIUM_DATA/local_inputs_202610/qrels_post_cutoff" --out "$RERANK_OUT"
```

## tab:agent

Producers: run_agent_eval.py, tau_stratified_agent.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: remote_pull/agent_n100/*.

```bash
python scripts/tau_stratified_agent.py --sample "$SAMPLE" --indices "$INDICES" --runs "$RUNS" --out "$OUT/agent.json"
# Optional agent rerun (makes model API calls):
python scripts/run_agent_eval.py --sample "$SAMPLE" --out-dir "$RUNS" --model "$AGENT_MODEL"
```

## tab:agent-wrapper

Producers: run_agent_eval_wrapper.py, coreb_search.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: remote_pull/agent_n100/wrapper_*.

```bash
python scripts/coreb_search.py build --repo "$CLONE" --index-dir "$INDEX" --mode both
python scripts/run_agent_eval_wrapper.py --sample "$SAMPLE" --index-root "$INDICES" --out-dir "$RUNS" --wrapper-mode dense --model "$AGENT_MODEL"
# Repeat with --wrapper-mode grep and router in separate output directories.
```

## tab:external

TODO: `audit/task_manifest.jsonl` is absent from the available development cache;
release or reconstruct the JSONL task metadata before running the aggregator.
The tau tool takes the separate `audit/eligible_main.tsv` manifest.

Producers: deepswe/aggregate_deepswe.py, deepswe/tau_top1_deepswe.py.

Cached inputs beneath `REPO_PREMIUM_DATA`: deepswe_pull/.

```bash
python scripts/deepswe/aggregate_deepswe.py --retrieval-root "$REPO_PREMIUM_DATA/deepswe_pull/retrieval" --manifest "$REPO_PREMIUM_DATA/deepswe_pull/audit/task_manifest.jsonl" --out "$OUT/deepswe.json"
# Optional tau rebuild (requires matching corpus snapshots):
python scripts/deepswe/tau_top1_deepswe.py --manifest "$REPO_PREMIUM_DATA/deepswe_pull/audit/eligible_main.tsv" --smoke "$RELEASE_ROOT/scripts" --out "$OUT/tau_top1_deepswe.jsonl"
```

## mediation numbers

Producers: mediation_decomposition.py (canonical c1000 config).

Cached inputs beneath `REPO_PREMIUM_DATA`: smoke_test_results/canon202607, tau_variants_c1000.

```bash
python scripts/mediation_decomposition.py --pull "$REPO_PREMIUM_DATA/smoke_test_results" --cell-a canon202607/jina05_cellA_c1000 --cell-b canon202607/jina05_cellB_c1000 --tau tau_variants_c1000 --out "$OUT/mediation.json"
```

