#!/usr/bin/env python3
"""In-place patcher: add reranker support to smoke_test_django.py.

Edits are string-based and idempotent (no-op if already applied).
"""
from pathlib import Path
import sys

TARGET = Path(str(Path(__file__).resolve().parents[1] / "run_retrieval.py"))
src = TARGET.read_text()

if "--rerank-model" in src:
    print("[patch] already applied, no-op")
    sys.exit(0)

# --- 1) Add rerank args to argparser ---
args_anchor = (
    "    p.add_argument(\"--tokenize-camelcase\", action=\"store_true\",\n"
    "                   help=\"Split camelCase identifiers into subwords in tokenize() \"\n"
    "                       \"(for JS/TS/Java/Go corpora). Off by default: preserves \"\n"
    "                       \"existing Python paper numbers.\")\n"
)
args_insert = args_anchor + (
    "    p.add_argument(\"--rerank-model\", default=\"\",\n"
    "                   help=\"HF model id for cross-encoder reranker (e.g. hq-bench/coreb-code-reranker). \"\n"
    "                        \"If empty, no reranking is performed.\")\n"
    "    p.add_argument(\"--rerank-topk\", type=int, default=50,\n"
    "                   help=\"Number of dense-top candidates to feed into reranker.\")\n"
    "    p.add_argument(\"--rerank-task\", default=\"text2code\",\n"
    "                   choices=[\"text2code\", \"code2code\", \"code2text\"],\n"
    "                   help=\"Task template for the reranker prompt (coreb-code-reranker convention).\")\n"
)
if args_anchor not in src:
    raise SystemExit("[patch] anchor for args not found; abort")
src = src.replace(args_anchor, args_insert, 1)

# --- 2) Extend QueryResult dataclass ---
qr_anchor = (
    "    dense_recall10: float\n"
    "    delta_ndcg10: float  # dense - bm25\n"
)
qr_insert = qr_anchor + (
    "    # reranker fields (defaults so old paths still work if reranker off)\n"
    "    rerank_top10: list[str] = None\n"
    "    rerank_ndcg10: float = -1.0\n"
    "    rerank_recall10: float = -1.0\n"
    "    delta_rerank_ndcg10: float = 0.0   # rerank - dense\n"
)
if qr_anchor not in src:
    raise SystemExit("[patch] anchor for QueryResult not found; abort")
src = src.replace(qr_anchor, qr_insert, 1)

# --- 3) Inject reranker loader helper before main() ---
loader_helper = '''
def _load_reranker(model_id: str, task: str):
    """Load hq-bench/coreb-code-reranker style CausalLM cross-encoder.

    Returns (score_fn, cleanup). score_fn takes (query, document) -> float.
    Uses logit(yes)-logit(no) at last position.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    task_map = {
        "text2code": "Given a natural language programming task, retrieve code that correctly solves or implements the task.",
        "code2code": "Given a code snippet, retrieve code that is semantically equivalent or solves the same task.",
        "code2text": "Given a code snippet, retrieve the natural language description or problem statement that best matches the code.",
    }
    instruct = task_map[task]

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    yes_id = tok.convert_tokens_to_ids("yes")
    no_id = tok.convert_tokens_to_ids("no")
    PREFIX = ("<|im_start|>system\\nJudge whether the Document meets the requirements "
              "based on the Query and the Instruct provided. Note that the answer "
              "can only be \\"yes\\" or \\"no\\".<|im_end|>\\n<|im_start|>user\\n")
    SUFFIX = "<|im_end|>\\n<|im_start|>assistant\\n"

    @torch.no_grad()
    def score(query: str, document: str) -> float:
        prompt = f"{PREFIX}<Instruct>: {instruct}\\n<Query>: {query}\\n<Document>: {document}{SUFFIX}"
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        logits = model(**inputs).logits[0, -1, :]
        return float((logits[yes_id] - logits[no_id]).item())

    return score

'''
main_anchor = "def main() -> int:\n"
if main_anchor not in src:
    raise SystemExit("[patch] main() anchor missing")
src = src.replace(main_anchor, loader_helper + main_anchor, 1)

# --- 4) After args parsing block ("args = p.parse_args()"), init reranker (once) ---
# We'll inject just after the "if args.tokenize_camelcase" block by replacing the "exts = tuple(...)" line.
init_anchor = "    exts = tuple(e.strip() for e in args.code_exts.split(\",\") if e.strip())\n"
init_insert = init_anchor + (
    "\n"
    "    rerank_score = None\n"
    "    if args.rerank_model:\n"
    "        print(f\"[rerank] loading {args.rerank_model} (task={args.rerank_task}, topk={args.rerank_topk})\", file=sys.stderr)\n"
    "        rerank_score = _load_reranker(args.rerank_model, args.rerank_task)\n"
)
if init_anchor not in src:
    raise SystemExit("[patch] init anchor missing")
src = src.replace(init_anchor, init_insert, 1)

# --- 5) After dense_top computation, rerank top-N if enabled ---
# Target block: "        order = np.argsort(-file_sims)[:args.k]\n        dense_top = [paths[i] for i in order]\n"
per_query_anchor = (
    "        order = np.argsort(-file_sims)[:args.k]\n"
    "        dense_top = [paths[i] for i in order]\n\n"
    "        grep_ndcg = ndcg_at_k(grep_top, gold, args.k)\n"
    "        bm25_ndcg = ndcg_at_k(bm25_top, gold, args.k)\n"
    "        dense_ndcg = ndcg_at_k(dense_top, gold, args.k)\n"
)
per_query_insert = (
    "        order = np.argsort(-file_sims)[:args.k]\n"
    "        dense_top = [paths[i] for i in order]\n\n"
    "        # Rerank if enabled: take dense-top-K candidates, rerank, take top args.k\n"
    "        rerank_top: list[str] = []\n"
    "        rerank_ndcg = -1.0\n"
    "        rerank_recall = -1.0\n"
    "        if rerank_score is not None:\n"
    "            # dense-top-K candidate FILES (not chunks) — reuse file_sims order\n"
    "            cand_order = np.argsort(-file_sims)[:args.rerank_topk]\n"
    "            cand_paths = [paths[i] for i in cand_order]\n"
    "            # Score each candidate: document = full file text (truncated by tokenizer)\n"
    "            scores = []\n"
    "            for cp in cand_paths:\n"
    "                doc = read_file(args.clone_dir, cp)[:MAX_FILE_CHARS]\n"
    "                scores.append(rerank_score(query, doc))\n"
    "            rr_order = np.argsort(-np.asarray(scores))[:args.k]\n"
    "            rerank_top = [cand_paths[j] for j in rr_order]\n"
    "            rerank_ndcg = ndcg_at_k(rerank_top, gold, args.k)\n"
    "            rerank_recall = recall_at_k(rerank_top, gold, args.k)\n\n"
    "        grep_ndcg = ndcg_at_k(grep_top, gold, args.k)\n"
    "        bm25_ndcg = ndcg_at_k(bm25_top, gold, args.k)\n"
    "        dense_ndcg = ndcg_at_k(dense_top, gold, args.k)\n"
)
if per_query_anchor not in src:
    raise SystemExit("[patch] per-query anchor missing")
src = src.replace(per_query_anchor, per_query_insert, 1)

# --- 6) Populate rerank_* in QueryResult constructor ---
qr_ctor_anchor = (
    "            delta_ndcg10=dense_ndcg - bm25_ndcg,\n"
    "        )\n"
)
qr_ctor_insert = (
    "            delta_ndcg10=dense_ndcg - bm25_ndcg,\n"
    "            rerank_top10=rerank_top,\n"
    "            rerank_ndcg10=rerank_ndcg,\n"
    "            rerank_recall10=rerank_recall,\n"
    "            delta_rerank_ndcg10=(rerank_ndcg - dense_ndcg) if rerank_score is not None else 0.0,\n"
    "        )\n"
)
if qr_ctor_anchor not in src:
    raise SystemExit("[patch] QueryResult ctor anchor missing")
src = src.replace(qr_ctor_anchor, qr_ctor_insert, 1)

# --- 7) Extend summary dict + print block ---
summary_anchor = (
    "        \"recall10\": {\n"
    "            \"grep\": mean([r.grep_recall10 for r in results]),\n"
    "            \"bm25\": mean([r.bm25_recall10 for r in results]),\n"
    "            \"dense\": mean([r.dense_recall10 for r in results]),\n"
    "        },\n"
    "    }\n"
)
summary_insert = (
    "        \"recall10\": {\n"
    "            \"grep\": mean([r.grep_recall10 for r in results]),\n"
    "            \"bm25\": mean([r.bm25_recall10 for r in results]),\n"
    "            \"dense\": mean([r.dense_recall10 for r in results]),\n"
    "        },\n"
    "    }\n"
    "    if any(r.rerank_ndcg10 >= 0 for r in results):\n"
    "        rr = [r for r in results if r.rerank_ndcg10 >= 0]\n"
    "        summary[\"rerank_model\"] = args.rerank_model\n"
    "        summary[\"rerank_topk\"] = args.rerank_topk\n"
    "        summary[\"ndcg10\"][\"rerank\"] = sum(r.rerank_ndcg10 for r in rr) / len(rr)\n"
    "        summary[\"ndcg10\"][\"premium_rerank_minus_dense\"] = sum(r.delta_rerank_ndcg10 for r in rr) / len(rr)\n"
    "        summary[\"recall10\"][\"rerank\"] = sum(r.rerank_recall10 for r in rr) / len(rr)\n"
)
if summary_anchor not in src:
    raise SystemExit("[patch] summary anchor missing")
src = src.replace(summary_anchor, summary_insert, 1)

# --- 8) Print loop: include rerank row if present ---
print_anchor = (
    "    for name in [\"grep\", \"bm25\", \"dense\"]:\n"
)
print_insert = (
    "    row_names = [\"grep\", \"bm25\", \"dense\"]\n"
    "    if \"rerank\" in summary[\"ndcg10\"]:\n"
    "        row_names.append(\"rerank\")\n"
    "    for name in row_names:\n"
)
if print_anchor not in src:
    raise SystemExit("[patch] print anchor missing")
src = src.replace(print_anchor, print_insert, 1)

TARGET.write_text(src)
print(f"[patch] applied; new size: {len(src.splitlines())} lines")
