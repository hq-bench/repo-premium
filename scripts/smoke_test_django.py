"""Backward-compatible alias for run_retrieval.

`import smoke_test_django` returns the run_retrieval module object itself (not a
star-import copy), so callers that set module globals such as
`TOKENIZE_CAMELCASE = True` change the flag the tokenizer actually reads.
"""
import sys

import run_retrieval

if __name__ == "__main__":
    raise SystemExit(run_retrieval.main())

sys.modules[__name__] = run_retrieval
