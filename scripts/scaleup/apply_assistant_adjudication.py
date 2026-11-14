#!/usr/bin/env python3
"""Apply versioned assistant-review labels without mutating Qwen outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REVISIONS: dict[str, tuple[str, str]] = {
    "microsoft__TypeScript-pr-63368": (
        "ATA package-name filtering allows malformed specifiers to reach npm, "
        "causing automatic type acquisition requests to fail. Reject invalid "
        "package-name specifiers before they are passed to npm.",
        "Removed the unsupported security-exploit implication; the raw PR says "
        "the maximum consequence is a failed ATA request.",
    ),
    "pandas-dev__pandas-pr-58056": (
        "Deprecate the behavior in pd.concat that ignores empty entries.",
        "Corrected semantic direction: DEPR requests deprecation, not new "
        "silent filtering behavior.",
    ),
    "gin-gonic__gin-pr-3834": (
        "Update the CI lint tooling and workflows for consistency, including "
        "upgrading the workflow tool from v1.52.2 to v1.55.2 and fixing the "
        "setup-go cache error.",
        "Restored the tool-version update omitted by the generated query.",
    ),
    "pandas-dev__pandas-pr-66655": (
        "Correct a batch of pandas user-guide and docstring inaccuracies; "
        "document existing behavior across parsing, I/O, time series, nullable "
        "arrays, and floating-point conversion; and add the requested cookbook "
        "examples and API usage notes.",
        "Restored the cookbook and general documentation scope omitted by the "
        "generated query while avoiding a solution-level list.",
    ),
}

KEEP_CORRECTIONS: dict[str, tuple[bool, str]] = {
    "apache__kafka-pr-23006": (
        True,
        "The minimal query is maximally faithful; prompt v2 permits an "
        "underspecified source without invented context.",
    ),
    "eslint__eslint-pr-21166": (
        False,
        "Validator false negative: its final rationale cites a parent-node AST "
        "strategy that is absent from the repaired candidate.",
    ),
    "apache__kafka-pr-15570": (
        True,
        "The backport source gives no behavior beyond the concise fix title.",
    ),
}

REPLACEMENTS: dict[str, tuple[str, str]] = {
    "microsoft__TypeScript-pr-63372": (
        "duplicate_backport",
        "Automated release-6.0 cherry-pick of selected PR 63368 with identical "
        "gold paths and reference-change size.",
    ),
    "elastic__elasticsearch-pr-156202": (
        "duplicate_backport",
        "Second selected branch backport of PR 156191; duplicates selected PR "
        "156201 with the same gold file.",
    ),
    "python-poetry__poetry-pr-9218": (
        "duplicate_backport",
        "1.8 backport of selected PR 9207 with identical gold paths and "
        "reference-change size.",
    ),
    "rust-lang__cargo-pr-13654": (
        "duplicate_backport",
        "Stable-1.77 backport of selected PR 13630 with the same task and gold "
        "file.",
    ),
    "tauri-apps__tauri-pr-9166": (
        "duplicate_issue",
        "Second selected implementation of the same basename-suffix bug and "
        "issue as PR 9165, with overlapping gold paths.",
    ),
    "axios__axios-pr-6199": (
        "release_generated_artifacts",
        "Release PR whose main gold is dominated by generated distribution "
        "bundles rather than the release-note bug fixes.",
    ),
    "axios__axios-pr-6204": (
        "release_generated_artifacts",
        "Release PR whose main gold is dominated by the same generated "
        "distribution bundles as adjacent release PR 6199.",
    ),
}

# Full raw text was inspected for the original five failures and the suspicious
# subset selected by backport, direction, omission, or implementation-detail
# screens. Remaining queue items received title/curated screening.
FULL_RAW_REVIEWED = {
    "apache__kafka-pr-15570",
    "apache__kafka-pr-15597",
    "apache__kafka-pr-23006",
    "axios__axios-pr-6307",
    "clap-rs__clap-pr-5344",
    "elastic__elasticsearch-pr-156199",
    "elastic__elasticsearch-pr-156202",
    "eslint__eslint-pr-21166",
    "gin-gonic__gin-pr-3834",
    "microsoft__TypeScript-pr-63368",
    "microsoft__TypeScript-pr-63372",
    "pandas-dev__pandas-pr-58056",
    "pandas-dev__pandas-pr-66655",
    "prometheus__prometheus-pr-13851",
    "pydantic__pydantic-pr-13584",
    "scipy__scipy-pr-25844",
    "vitest-dev__vitest-pr-10842",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit-queue", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    records = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    queue_ids = {
        json.loads(line)["task_id"]
        for line in args.audit_queue.read_text().splitlines()
        if line.strip()
    }
    output: list[dict[str, Any]] = []
    for original in records:
        row = dict(original)
        task_id = row["task_id"]
        review_scope = (
            "full_raw"
            if task_id in FULL_RAW_REVIEWED
            else "title_curated_screen"
            if task_id in queue_ids
            else "automated_checks_only"
        )
        review: dict[str, Any] = {
            "reviewer_type": "assistant_model",
            "human_verified": False,
            "review_scope": review_scope,
            "action": "keep",
            "decision": (
                "pass"
                if review_scope == "full_raw"
                else "screen_pass"
                if review_scope == "title_curated_screen"
                else "not_reviewed"
            ),
            "reason": "No issue found in the assigned review scope.",
        }
        row["assistant_curated_query"] = row["curated_query"]
        row["assistant_source_underspecified"] = row["final_audit"].get(
            "source_underspecified", False
        )
        if task_id in KEEP_CORRECTIONS:
            underspecified, reason = KEEP_CORRECTIONS[task_id]
            review.update(decision="pass", reason=reason)
            row["assistant_source_underspecified"] = underspecified
        if task_id in REVISIONS:
            query, reason = REVISIONS[task_id]
            review.update(action="revise", decision="pass", reason=reason)
            row["assistant_curated_query"] = query
        if task_id in REPLACEMENTS:
            category, reason = REPLACEMENTS[task_id]
            review.update(
                action="replace",
                decision="reject_task",
                category=category,
                reason=reason,
            )
        row["assistant_adjudication"] = review
        output.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "curated_queries_assistant_adjudicated.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output)
    )
    kept = [row for row in output if row["assistant_adjudication"]["action"] != "replace"]
    (args.out_dir / "curated_queries_kept_pending_replacements.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept)
    )
    summary = {
        "n_records": len(output),
        "n_audit_queue_screened": len(queue_ids),
        "n_full_raw_reviewed": sum(
            row["assistant_adjudication"]["review_scope"] == "full_raw"
            for row in output
        ),
        "n_revised": sum(
            row["assistant_adjudication"]["action"] == "revise" for row in output
        ),
        "n_replace": sum(
            row["assistant_adjudication"]["action"] == "replace" for row in output
        ),
        "n_kept_pending_replacements": len(kept),
        "action_counts": dict(
            Counter(row["assistant_adjudication"]["action"] for row in output)
        ),
        "human_verification_complete": False,
    }
    (args.out_dir / "assistant_adjudication_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
