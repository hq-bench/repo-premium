"""embedding_premium.Router — per-query regime predictor for code search.

Implements the deployment-time τ-router that operationalises F1 (the
embedding-premium regularity) and the three-regime framing from
coreb_repo_paper.tex §4.1 and §7.

Inputs:
  - a corpus (list of chunks, or a directory of files chunked with
    `smoke_test_django.chunk_file` at c=1000) used to build a per-corpus
    IDF table at fit time.

Outputs (per query):
  - tau_query(q): top-K mean IDF of the stripped query's tokens against
    the corpus IDF table (K=5 by default, matching §5.3 R3 ablation
    `tau_mean5`). This is the deployable proxy for the offline τ in
    Eq.(1) of the paper, which requires the gold file. The top-K mean
    captures the rarest, most distinctive query tokens; on the §5.3
    pooled τ>0 regime it tracks τ_jaccard quartile boundaries closely
    (low → mean_idf < 6, high → mean_idf > 7).
  - has_identifiers(q): True iff the query contains at least one
    CamelCase, snake_case, dotted-path, or backticked symbol.
  - predict(q): one of {"lexical", "structural", "semantic", "hybrid"}
    per the §7 decision rule.

Default thresholds are taken from the §5.3 pooled τ>0 quartile
boundaries. Users can recalibrate via `Router.fit(...).calibrate(...)`.

Usage:
    from embedding_premium import Router

    chunks = [open(p).read() for p in glob.glob("repo/**/*.py")]
    router = Router.fit(chunks)
    regime = router.predict("compute the modular inverse without mutation")
    # → "semantic"

This module is CPU-only and has no torch / transformers dependency.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Reuse the paper-canonical tokenize and boilerplate stripper so that
# router-time τ is computed against the same token distribution used
# offline in §5.3.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from run_retrieval import (  # noqa: E402
    compute_idf,
    strip_pr_boilerplate,
    tokenize,
)


# --- identifier detection -------------------------------------------------
# CamelCase: 2+ capital letters with lowercase between (e.g. ParseError,
# HTTPClient). snake_case: token with an internal underscore not at the
# ends. Dotted path: a.b.c with at least one dot between identifiers.
# Backtick: any text between backticks.
_CAMEL_RE = re.compile(r"\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b")
_SNAKE_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b")
_DOTTED_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_BACKTICK_RE = re.compile(r"`[^`\n]+`")


def has_identifiers(text: str) -> bool:
    """True iff the (stripped) text contains any code-like identifier.

    Used by Router.predict to decide structural- vs semantic-regime routing.
    """
    return bool(
        _CAMEL_RE.search(text)
        or _SNAKE_RE.search(text)
        or _DOTTED_RE.search(text)
        or _BACKTICK_RE.search(text)
    )


# --- default thresholds ---------------------------------------------------
# Data-driven thresholds from the §6 R-ROUTER deployment-time validation
# (validate_router.py on Cell B, jina-code-0.5b, stripped, c=1000).
# Cell B is the regime where F1 is strongest and where the deployment
# proxy recovers a strictly monotone Δ-by-quartile signal (Pearson r=-0.12,
# Q1→Q4 means +0.276 → +0.151).
#
# Empirical Cell B quartile boundaries on the router's top-K mean-IDF:
#   Q1/Q2 = 8.23, Q2/Q3 = 8.84, Q3/Q4 = 9.45
# Even Q4 still has Δ > 0 (dense beats BM25 by +0.15), so we never route
# to "lexical alone"; we either route to dense or to a hybrid blend.
DEFAULT_HYBRID_THRESHOLD = 9.45    # top-K mean IDF >= this  → hybrid blend
DEFAULT_SEMANTIC_THRESHOLD = 8.23  # top-K mean IDF <  this  → semantic / dense
DEFAULT_TOP_K = 5                   # matches paper R3 baseline tau_mean5


@dataclass
class Router:
    """Per-corpus query-regime predictor.

    Build with Router.fit(corpus_chunks). The returned instance is
    callable: regime = router.predict(query).
    """
    idf: dict[str, float] = field(default_factory=dict)
    n_chunks: int = 0
    hybrid_threshold: float = DEFAULT_HYBRID_THRESHOLD
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD
    top_k: int = DEFAULT_TOP_K
    strip_boilerplate: bool = True

    # -- construction ------------------------------------------------------
    @classmethod
    def fit(
        cls,
        corpus_chunks: Iterable[str],
        *,
        hybrid_threshold: float = DEFAULT_HYBRID_THRESHOLD,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        strip_boilerplate: bool = True,
    ) -> "Router":
        """Build the per-corpus IDF table from an iterable of chunk strings.

        `corpus_chunks` should be the same chunks (c=1000 with 10-line
        overlap) used for retrieval, so that the router's IDF distribution
        matches what BM25 / dense see.
        """
        doc_tokens = [tokenize(c) for c in corpus_chunks]
        idf = compute_idf(doc_tokens)
        return cls(
            idf=idf,
            n_chunks=len(doc_tokens),
            hybrid_threshold=hybrid_threshold,
            semantic_threshold=semantic_threshold,
            top_k=top_k,
            strip_boilerplate=strip_boilerplate,
        )

    # -- per-query measurements --------------------------------------------
    def _prep(self, query: str) -> tuple[str, list[str]]:
        text = strip_pr_boilerplate(query) if self.strip_boilerplate else query
        return text, tokenize(text)

    def tau_query(self, query: str) -> float:
        """Top-K mean IDF of the stripped query's tokens (deployment-time
        proxy for offline τ_jaccard, matching the §5.3 R3 baseline
        tau_mean5). Returns 0.0 on an empty token set, which the predictor
        treats as the degenerate regime.

        Why top-K and not full mean: short, distinctive query tokens
        (CamelCase identifiers, dotted paths) carry most of the
        discriminative signal; averaging over all tokens including
        natural-language stopwords ('the', 'fix', 'when') collapses the
        scale by ~2× and breaks the §5.3 thresholds.
        """
        _, toks = self._prep(query)
        if not toks:
            return 0.0
        vals = sorted((self.idf.get(t, 0.0) for t in toks), reverse=True)
        top = vals[: self.top_k]
        return float(sum(top) / len(top)) if top else 0.0

    def has_identifiers(self, query: str) -> bool:
        text, _ = self._prep(query)
        return has_identifiers(text)

    # -- routing rule -------------------------------------------------------
    def predict(self, query: str) -> str:
        """Return one of {"structural", "semantic", "hybrid"}.

        Note: there is no "lexical alone" regime in the §6 R-ROUTER
        validation data — even at the highest router-IDF quartile
        (Cell B Q4, mean top-K IDF $\\geq$ 9.45), dense still beats BM25 by
        $\\sim$0.15 nDCG@10. The router therefore never emits "lexical"
        in the default configuration; the highest-IDF queries are routed
        to a BM25-dominated hybrid blend, not to BM25 alone.

        Rule (matches §7 of the paper):
          - mean_idf <  semantic_threshold and identifiers
                                                → "structural" (dense + graph rerank)
          - mean_idf <  semantic_threshold      → "semantic"   (dense alone)
          - else (mean_idf >= semantic_threshold)
                                                → "hybrid"     (α-blend dense + BM25)
        """
        m = self.tau_query(query)
        if m < self.semantic_threshold:
            return "structural" if self.has_identifiers(query) else "semantic"
        return "hybrid"

    def alpha_blend(self, query: str) -> float:
        """Return α in [0, 1] = fraction of weight on the BM25 score in
        an α-blended RRF / score-sum combination with the dense score.

        Interpolates linearly between (semantic_threshold → α=0, pure dense)
        and (hybrid_threshold → α=0.5, balanced). Saturates at α=0.5 above
        hybrid_threshold (we never weight BM25 above dense, since even on
        the high-IDF Cell B Q4 dense still wins by +0.15 nDCG@10).

        This is the continuous routing signal used by the hybrid retriever;
        predict() is the categorical readout for logging and analysis.
        """
        m = self.tau_query(query)
        if m <= self.semantic_threshold:
            return 0.0
        if m >= self.hybrid_threshold:
            return 0.5
        # linear interpolation between the two thresholds
        span = self.hybrid_threshold - self.semantic_threshold
        return 0.5 * (m - self.semantic_threshold) / span

    def explain(self, query: str) -> dict:
        """Diagnostics dict — useful for ablation, logging, and the §5.3
        round-trip validation script."""
        text, toks = self._prep(query)
        m = self.tau_query(query)
        ids = has_identifiers(text)
        return {
            "stripped": text,
            "n_tokens": len(toks),
            "n_known": sum(1 for t in toks if t in self.idf),
            "mean_idf": m,
            "has_identifiers": ids,
            "regime": self.predict(query),
            "alpha_bm25": self.alpha_blend(query),
            "thresholds": (self.semantic_threshold, self.hybrid_threshold),
        }

    # -- calibration --------------------------------------------------------
    def calibrate(
        self,
        per_query_idf_and_delta: list[tuple[float, float]],
    ) -> "Router":
        """Re-fit (semantic, hybrid) thresholds from labelled (mean_idf, Δ)
        pairs by quartile boundaries. Mutates self in place and returns it.

        Intended for users with their own per-corpus router-validation set;
        the paper's default thresholds come from the §6 R-ROUTER Cell B
        quartile boundaries (jina-code-0.5b, c=1000, stripped).
        """
        if not per_query_idf_and_delta:
            return self
        vals = sorted(m for m, _ in per_query_idf_and_delta)
        q1 = vals[len(vals) // 4]
        q3 = vals[3 * len(vals) // 4]
        self.semantic_threshold = float(q1)
        self.hybrid_threshold = float(q3)
        return self


# --- small sanity-check entrypoint ---------------------------------------
def _demo() -> int:
    """Sanity-check on a toy corpus + three illustrative queries."""
    toy_corpus = [
        "def parse_url(url): return urllib.parse.urlparse(url)",
        "class RecursionError(Exception): pass",
        "def lazy_property(fn): return cached_property(fn)",
        "def modular_inverse(a, m): return pow(a, -1, m)",
        "def validate_dependency(node): return node.parents",
    ]
    router = Router.fit(toy_corpus)
    queries = [
        ("fix RecursionError in lazy_property", "lexical/structural"),
        ("what calls validate_dependency?", "structural"),
        ("compute the modular inverse without mutation", "semantic"),
    ]
    print(f"n_chunks={router.n_chunks} "
          f"vocab={len(router.idf)} "
          f"thr=({router.semantic_threshold:.2f}, "
          f"{router.hybrid_threshold:.2f})")
    for q, expected in queries:
        info = router.explain(q)
        print(f"  q={q!r}\n    "
              f"mean_idf={info['mean_idf']:.3f} "
              f"ids={info['has_identifiers']} "
              f"α_bm25={info['alpha_bm25']:.2f} "
              f"→ {info['regime']:<10s} (expected: {expected})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
