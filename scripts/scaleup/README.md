# Matched scale-up tooling

Mine the two natural PR windows into separate immutable directories:

```bash
python3 scripts/scaleup/mine_prs.py \
  --repos scripts/scaleup/repos.json \
  --merged-start 2023-01-01 --merged-end 2024-03-31 \
  --max-per-repo 25 --out-dir ${REPO_PREMIUM_DATA:-data}/scaleup/raw_pre

python3 scripts/scaleup/mine_prs.py \
  --repos scripts/scaleup/repos.json \
  --merged-start 2026-01-01 \
  --max-per-repo 25 --out-dir ${REPO_PREMIUM_DATA:-data}/scaleup/raw_post
```

The miner records exact base commits and distinguishes the reference patch from
the subset of production-code paths visible in the base snapshot. It does not
clone repositories or generate curated text.

Select the deterministic matched core after both windows finish:

```bash
python3 scripts/scaleup/select_balanced.py \
  --pre ${REPO_PREMIUM_DATA:-data}/scaleup/raw_pre/candidates.jsonl \
        ${REPO_PREMIUM_DATA:-data}/scaleup/reserve_pre/candidates.jsonl \
  --post ${REPO_PREMIUM_DATA:-data}/scaleup/raw_post/candidates.jsonl \
         ${REPO_PREMIUM_DATA:-data}/scaleup/reserve_post/candidates.jsonl \
  --per-window-repo 5 \
  --out-dir ${REPO_PREMIUM_DATA:-data}/scaleup/balanced
```

Repositories lacking five eligible tasks in either window are excluded from
the matched headline core but retained in the raw candidate pool.

Before cloning corpora or launching retrieval, verify every selected gold path
against its exact base commit from an authenticated cluster login:

```bash
python3 scripts/scaleup/verify_base_paths.py \
  --input ${REPO_PREMIUM_DATA:-data}/scaleup/balanced/balanced_tasks.jsonl \
  --out-dir ${REPO_PREMIUM_DATA:-data}/scaleup/verified
```

If any task fails, do not drop it in isolation: replace it from the same
repository/window candidate pool and rerun deterministic selection, or exclude
the whole repository to preserve the 5+5 matched design.

The cluster compute-node image does not contain `gh`, so run this lightweight
network audit from the authenticated cluster login node rather than through
Slurm. For a disconnect-safe run:

```bash
mkdir -p ${REPO_PREMIUM_DATA:-data}/scaleup/logs
nohup python3 scripts/scaleup/verify_base_paths.py \
  --input ${REPO_PREMIUM_DATA:-data}/scaleup/balanced/balanced_tasks.jsonl \
  --out-dir ${REPO_PREMIUM_DATA:-data}/scaleup/verified \
  --workers 8 \
  > ${REPO_PREMIUM_DATA:-data}/scaleup/logs/verify_login.out 2>&1 \
  < /dev/null &
```

Generate a balanced 20-task Qwen-122B curation/verification pilot (two tasks
per language x date-window cell):

```bash
# Set OPENAI_API_BASE and OPENAI_API_KEY in the environment first.
python3 scripts/scaleup/curate_and_verify.py \
  --input ${REPO_PREMIUM_DATA:-data}/scaleup/verified/verified_tasks.jsonl \
  --out-dir ${REPO_PREMIUM_DATA:-data}/scaleup/curation/qwen122b_pilot \
  --model qwen35-122B \
  --pilot-per-language-window 2
```

The generator never receives gold paths or patch content. It removes submitted
patch mechanics even when the raw PR narrates them, while preserving public API
names needed to state the task. Validation is a separate blinded call, and one
repair/revalidation attempt is allowed. Sparse source text is flagged as
`source_underspecified` rather than being expanded with invented facts. Outputs
are labelled `llm_verified`; the pilot's `human_audit_queue.jsonl` contains all
20 tasks and must be reviewed before anything is labelled `human_verified`.

Apply the versioned assistant-review corrections without overwriting model
outputs:

```bash
python3 scripts/scaleup/apply_assistant_adjudication.py \
  --input /path/to/qwen122b_full_v2/curated_queries.jsonl \
  --audit-queue /path/to/qwen122b_full_v2/human_audit_queue.jsonl \
  --out-dir /path/to/assistant_adjudicated_v1
```

`assistant_adjudication.action=replace` records must be replaced from the same
repository/date cell before analysis. Assistant review never changes
`human_verified` to true.

Build a balanced clean manifest after extending the candidate pool for any
under-supplied repository/window cells:

```bash
python3 scripts/scaleup/build_clean_manifest.py \
  --selected /path/to/balanced/balanced_tasks.jsonl \
  --adjudicated /path/to/assistant_adjudicated_v1/curated_queries_assistant_adjudicated.jsonl \
  --pre /path/to/raw_pre/candidates.jsonl /path/to/replacement_pre/candidates.jsonl \
  --post /path/to/raw_post/candidates.jsonl /path/to/replacement_post/candidates.jsonl \
  --out-dir /path/to/clean
```

Verify and curate every replacement before merging. Then create the flattened,
authoritative analysis label file without mutating any source artifact:

```bash
python3 scripts/scaleup/merge_clean_labels.py \
  --manifest /path/to/clean/clean_balanced_tasks.jsonl \
  --adjudicated /path/to/assistant_adjudicated_v1/curated_queries_assistant_adjudicated.jsonl \
  --replacement-labels /path/to/replacement_curation/curated_queries.jsonl \
  --original-manifest /path/to/balanced/balanced_tasks.jsonl \
  --out-dir /path/to/clean/final_labels
```

`coreb_scaleup_clean_labels_v1.jsonl` is the retrieval input. Each row contains
the authoritative raw and curated query, main gold files, label-quality scope,
and replacement provenance. `human_verified` remains false until an actual
human audit is completed.

Freeze the repository-disjoint development/test assignment before fitting any
router or fusion parameters:

```bash
python3 scripts/scaleup/freeze_splits.py \
  --labels /path/to/final_labels/coreb_scaleup_clean_labels_v1.jsonl \
  --manifest /path/to/clean_balanced_tasks.jsonl \
  --out-dir /path/to/clean/splits
```

The default seed is `20260810`; development receives approximately one third
of repositories within each language. Every repository contributes all ten of
its tasks to exactly one partition.

Acquire objects, then fail closed on corpus integrity. Fetching writes a few
large packfiles per repository, which the shared filesystem handles well. No
working tree is materialized at this stage.

```bash
python3 scripts/scaleup/fetch_mirrors.py \
  --manifest /path/to/clean/splits/retrieval_tasks.jsonl \
  --mirror-root /path/to/experiment/mirrors_full \
  --workers 6

python3 scripts/scaleup/audit_corpora.py \
  --tasks /path/to/clean/splits/retrieval_tasks.jsonl \
  --mirror-root /path/to/experiment/mirrors_full \
  --out-dir /path/to/experiment/audit
```

`fetch_mirrors.py` is resumable: commits already present are skipped. The audit
reads each task's file list out of the bare mirror with `git ls-tree` at the
exact base commit, so it costs one pack traversal per task and materializes
nothing. Reading by SHA is a stronger exactness guarantee than comparing a
checkout's `HEAD`, because there is no working tree that could have drifted.
The audit refuses eligibility if any main gold path is absent from the
enumerated corpus, and writes two qrel records per task, one raw and one
curated, so the retrieval job encodes the corpus once for the paired
comparison. Retrieval jobs must consume only the generated `eligible_main.tsv`.

### Why the corpus is no longer checked out

The original design (`prepare_repos_shared.py`, kept for provenance) created
one detached worktree per task, 360 full repository trees on JuiceFS. Because
that filesystem charges roughly one metadata round-trip per file, a single
large-repository checkout took two to five hours, and the ten worktrees of one
repository were created serially. After 29.5 hours only 170/360 snapshots
existed, with every large repository still outstanding. The run was stopped.

`git ls-tree` for the audit and `git archive` into compute-node-local scratch
for retrieval do the same work without ever writing a tree to shared storage.
`verify_lstree_equivalence.py` proves the substitution is sound by comparing
both enumerations on every task that has a valid worktree *and* a mirror:

```bash
python3 scripts/scaleup/verify_lstree_equivalence.py \
  --tasks /path/to/clean/splits/retrieval_tasks.jsonl \
  --repo-root /path/to/experiment/repos \
  --mirror-root /path/to/experiment/mirrors_full \
  --out /path/to/experiment/audit/lstree_equivalence.jsonl
```

If you ever re-run the old worktree preparer, run `git worktree prune` in each
mirror first: interrupted targets are renamed into `quarantine_interrupted/`
but their registrations survive in the mirror's `worktrees/` admin directory,
and a re-add to the same path fails until they are pruned.

### Cluster execution note (2026-08-11)

The canonical active paths are:

```text
${REPO_PREMIUM_DATA:-data}/scaleup/experiment_v1/mirrors_full/
${REPO_PREMIUM_DATA:-data}/scaleup/experiment_v1/repos/
${REPO_PREMIUM_DATA:-data}/scaleup/experiment_v1/audit/
${REPO_PREMIUM_DATA:-data}/scaleup/experiment_v1/logs/
```

Worktree preparation was stopped after 29.5 hours with 170/360 snapshots from
17/36 repositories and zero recorded failures. Because the driver only writes
`prepare_shared_status.jsonl` after the whole pool drains, that file does not
exist for the stopped run; the authoritative record is the repository
completion lines in `logs/prepare_shared_full.out`, and readiness is anyway
re-derived by the corpus audit. Do not use checkout-directory count as
progress — it includes worktrees that were still being populated.

`mirrors_full/` survived and is the artifact that matters: the fetched packs
are the expensive-to-replace part, and the current flow needs nothing else.
The 201 checkout directories and 8 quarantined targets are retained only as
the baseline for `verify_lstree_equivalence.py`; they can be reclaimed once
that check has passed and retrieval has been validated end to end.

Eligibility remains the fail-closed 360/360 audit. Do not submit a partial
retrieval array.
