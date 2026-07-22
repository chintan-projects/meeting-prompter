"""Corpus preparation — distill raw docs into borrowable answer-units (F-701).

The product is retrieval-first (D-08): in a meeting we retrieve a span of the
user's corpus and show it verbatim — so corpus quality is the ceiling on output
quality. This package turns raw explainer docs into an answer-shaped corpus:

- :mod:`lib.corpus.text` — markdown → readable, borrowable prose.
- :mod:`lib.corpus.distiller` — reshape each section into grounded,
  provenance-tagged answer-units (heuristic + consent-gated cloud backends).

See docs/architecture/ADR-001-local-corpus-distiller.md and
docs/architecture/corpus-prep-onboarding-spec.md.
"""

from lib.corpus.distiller import distill
from lib.corpus.text import clean_markdown

__all__ = ["clean_markdown", "distill"]
