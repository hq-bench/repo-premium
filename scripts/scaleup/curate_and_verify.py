#!/usr/bin/env python3
"""Generate paired curated queries and independently verify them with LiteLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROMPT_VERSION = "coreb-scaleup-curation-v2"

GENERATOR_SYSTEM = """You edit software-engineering task descriptions for a code-search benchmark. Rewrite the supplied raw pull-request text into a concise, standalone problem statement that says WHAT behavior is wrong or requested, not HOW the submitted patch implements it. Preserve only requested behavior, observed symptoms, externally meaningful constraints, and acceptance criteria explicitly present in the raw text. Remove PR bookkeeping, review conversation, patch mechanics, implementation narration, chosen algorithms/data structures/control flow, test commands, rollout notes, links, issue numbers, commit hashes, and thanks. Identifiers may remain only when the raw text uses them to identify the affected public API, error, or observable component; never introduce a new identifier or file path. If the raw text is underspecified, keep the result underspecified rather than inventing context. Return JSON only: {\"curated_query\": \"...\"}."""

VALIDATOR_SYSTEM = """You are an independent benchmark annotation auditor. Compare a raw pull-request query with a proposed curated problem statement. You did not see how the candidate was generated. Apply every criterion strictly. Return JSON only with booleans semantic_fidelity, no_invented_requirements, no_new_solution_clues, implementation_neutral, standalone_clarity, pr_bookkeeping_removed, and source_underspecified; a decision equal to pass only when the first six criteria are true; reasons as a list of short strings; and repair_instructions as a short string. implementation_neutral is false when the candidate retains HOW the submitted patch works—specific control flow, algorithms, data structures, file edits, or patch strategy—even if those mechanics occur in the raw text. Preserve identifiers only when needed to name an affected public API, error, or observable component. standalone_clarity means the candidate uses all available raw context and does not depend on PR bookkeeping; when the raw source itself lacks context, set source_underspecified true but do not fail standalone_clarity merely for refusing to invent missing facts. New solution clues are facts absent from the raw query."""

REPAIR_SYSTEM = """Repair a proposed benchmark problem statement using the raw pull-request text and audit feedback. State the problem, requested behavior, and externally meaningful constraints, not the submitted patch mechanics. Preserve raw task semantics, remove unsupported content and bookkeeping, and do not introduce solution clues. If the source is sparse, remain sparse rather than inventing context. Return JSON only: {\"curated_query\": \"...\"}."""

BOOL_FIELDS = (
    "semantic_fidelity",
    "no_invented_requirements",
    "no_new_solution_clues",
    "implementation_neutral",
    "standalone_clarity",
    "pr_bookkeeping_removed",
)


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response is not a JSON object")
    return value


class LiteLLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int) -> None:
        import openai

        self.client = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout
        )
        self.model = model

    def json_call(self, system: str, user: str, max_tokens: int) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            # The proxy's Qwen reasoning mode can consume the full output
            # budget before emitting JSON. This is a deterministic annotation
            # task, so disable hidden reasoning through the vLLM chat template.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise ValueError(
                "model returned empty content "
                f"(finish_reason={response.choices[0].finish_reason})"
            )
        return parse_json_object(content)


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def stable_key(task_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def select_rows(
    rows: list[dict[str, Any]], pilot_per_language_window: int, seed: int
) -> list[dict[str, Any]]:
    if pilot_per_language_window <= 0:
        return sorted(rows, key=lambda row: row["task_id"])
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["language_group"], row["date_window"])].append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        ordered = sorted(
            groups[key], key=lambda row: stable_key(row["task_id"], seed)
        )
        selected.extend(ordered[:pilot_per_language_window])
    return sorted(selected, key=lambda row: row["task_id"])


def raw_text(row: dict[str, Any], max_chars: int) -> str:
    value = (row.get("raw_query") or "").strip()
    if not value:
        value = f"{row.get('title', '')}\n\n{row.get('body', '')}".strip()
    return value[:max_chars]


def generate(client: LiteLLMClient, raw: str, max_tokens: int) -> str:
    value = client.json_call(
        GENERATOR_SYSTEM,
        f"RAW PULL-REQUEST QUERY:\n<raw>\n{raw}\n</raw>",
        max_tokens,
    )
    curated = str(value.get("curated_query", "")).strip()
    if not curated:
        raise ValueError("generator returned an empty curated_query")
    return curated


def validate(
    client: LiteLLMClient, raw: str, curated: str, max_tokens: int
) -> dict[str, Any]:
    value = client.json_call(
        VALIDATOR_SYSTEM,
        f"RAW QUERY:\n<raw>\n{raw}\n</raw>\n\n"
        f"CANDIDATE:\n<candidate>\n{curated}\n</candidate>",
        max_tokens,
    )
    normalized = {field: value.get(field) is True for field in BOOL_FIELDS}
    normalized["decision"] = "pass" if all(normalized.values()) else "fail"
    reasons = value.get("reasons", [])
    normalized["reasons"] = reasons if isinstance(reasons, list) else [str(reasons)]
    normalized["repair_instructions"] = str(
        value.get("repair_instructions", "")
    ).strip()
    normalized["source_underspecified"] = value.get("source_underspecified") is True
    return normalized


def repair(
    client: LiteLLMClient,
    raw: str,
    curated: str,
    audit: dict[str, Any],
    max_tokens: int,
) -> str:
    value = client.json_call(
        REPAIR_SYSTEM,
        f"RAW QUERY:\n<raw>\n{raw}\n</raw>\n\n"
        f"CANDIDATE:\n<candidate>\n{curated}\n</candidate>\n\n"
        f"AUDIT FEEDBACK:\n{json.dumps(audit, ensure_ascii=False)}",
        max_tokens,
    )
    repaired = str(value.get("curated_query", "")).strip()
    if not repaired:
        raise ValueError("repair returned an empty curated_query")
    return repaired


def process_row(
    client: LiteLLMClient, row: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    raw = raw_text(row, args.max_input_chars)
    candidate = generate(client, raw, args.max_tokens)
    initial_audit = validate(client, raw, candidate, args.max_tokens)
    repaired = False
    final_candidate = candidate
    final_audit = initial_audit
    if initial_audit["decision"] == "fail" and args.repair_attempts:
        final_candidate = repair(
            client, raw, candidate, initial_audit, args.max_tokens
        )
        final_audit = validate(client, raw, final_candidate, args.max_tokens)
        repaired = True
    return {
        "task_id": row["task_id"],
        "repo": row["repo"],
        "language_group": row["language_group"],
        "date_window": row["date_window"],
        "raw_query": raw,
        "curated_query_initial": candidate,
        "initial_audit": initial_audit,
        "repair_attempted": repaired,
        "curated_query": final_candidate,
        "final_audit": final_audit,
        "llm_verified": final_audit["decision"] == "pass",
        "human_verified": False,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "prompt_hashes": {
            "generator": prompt_hash(GENERATOR_SYSTEM),
            "validator": prompt_hash(VALIDATOR_SYSTEM),
            "repair": prompt_hash(REPAIR_SYSTEM),
        },
    }


def write_aggregate(out_dir: Path, selected: list[dict[str, Any]]) -> None:
    records = []
    for row in selected:
        path = out_dir / "tasks" / f"{row['task_id']}.json"
        if path.exists():
            records.append(json.loads(path.read_text()))
    records.sort(key=lambda row: row["task_id"])
    (out_dir / "curated_queries.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    )
    final_pass = [row for row in records if row["llm_verified"]]
    summary = {
        "prompt_version": PROMPT_VERSION,
        "n_selected": len(selected),
        "n_completed": len(records),
        "n_initial_pass": sum(
            row["initial_audit"]["decision"] == "pass" for row in records
        ),
        "n_repaired": sum(row["repair_attempted"] for row in records),
        "n_final_pass": len(final_pass),
        "n_final_fail": len(records) - len(final_pass),
        "n_source_underspecified": sum(
            row["final_audit"].get("source_underspecified", False)
            for row in records
        ),
        "counts_by_language": dict(Counter(row["language_group"] for row in records)),
        "counts_by_window": dict(Counter(row["date_window"] for row in records)),
        "human_verification_complete": False,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # A pilot is fully audited by a human. For a full run, queue every failure
    # plus a deterministic 20% of passes, stratified upstream by the selected set.
    if len(selected) <= 20:
        audit_queue = records
    else:
        failures = [row for row in records if not row["llm_verified"]]
        pass_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in final_pass:
            pass_groups[(row["language_group"], row["date_window"])].append(row)
        sampled_passes: list[dict[str, Any]] = []
        for key in sorted(pass_groups):
            group = sorted(
                pass_groups[key],
                key=lambda row: stable_key(row["task_id"], 20260809),
            )
            sampled_passes.extend(group[: max(1, round(0.20 * len(group)))])
        audit_queue = failures + sampled_passes
        audit_queue.sort(key=lambda row: row["task_id"])
    (out_dir / "human_audit_queue.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_queue)
    )
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen35-122B")
    parser.add_argument("--api-url", default=os.environ.get("OPENAI_API_BASE"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--pilot-per-language-window", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--max-input-chars", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--repair-attempts", type=int, choices=(0, 1), default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if not args.api_url:
        parser.error("set OPENAI_API_BASE or pass --api-url")
    if not args.api_key:
        parser.error("set OPENAI_API_KEY or pass --api-key")

    rows = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("base_verified", True)]
    selected = select_rows(rows, args.pilot_per_language_window, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tasks").mkdir(exist_ok=True)
    client = LiteLLMClient(args.api_url, args.api_key, args.model, args.timeout)
    print(f"selected {len(selected)} tasks; model={args.model}", flush=True)
    for number, row in enumerate(selected, start=1):
        path = args.out_dir / "tasks" / f"{row['task_id']}.json"
        if path.exists():
            print(f"[{number}/{len(selected)}] {row['task_id']}: skip", flush=True)
            continue
        try:
            result = process_row(client, row, args)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(
                f"[{number}/{len(selected)}] {row['task_id']}: "
                f"{result['final_audit']['decision']}",
                flush=True,
            )
        except Exception as exc:
            print(f"[{number}/{len(selected)}] {row['task_id']}: ERROR {exc}", flush=True)
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)
    write_aggregate(args.out_dir, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
