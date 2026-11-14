# DeepSWE extension

Obtain the official task source from https://github.com/datacurve-ai/deep-swe
and place it under `$REPO_PREMIUM_DATA/deepswe/source`.

```bash
python scripts/deepswe/audit_tasks.py \
  --tasks-root "$REPO_PREMIUM_DATA/deepswe/source/tasks" \
  --out-dir "$REPO_PREMIUM_DATA/deepswe/audit"
python scripts/deepswe/prepare_repos.py \
  --manifest "$REPO_PREMIUM_DATA/deepswe/audit/task_manifest.jsonl" \
  --repo-root "$REPO_PREMIUM_DATA/deepswe/repos" --workers 8
```

Rerun `audit_tasks.py` with `--repo-root` to verify gold paths against the
checked-out base snapshots. Run `scripts/run_retrieval.py` per task using
its qrels and clone directory; choose an encoder with `--dense-model`.
`aggregate_deepswe.py` reads the cached per-task results and manifest:

```bash
python scripts/deepswe/aggregate_deepswe.py \
  --retrieval-root "$REPO_PREMIUM_DATA/deepswe_pull/retrieval" \
  --manifest "$REPO_PREMIUM_DATA/deepswe_pull/audit/task_manifest.jsonl" \
  --out deepswe_summary.json
python scripts/deepswe/tau_top1_deepswe.py --help
```

Cache filenames and the paper mapping still need verification against the
public dataset. Server-specific container images and launchers are excluded.
