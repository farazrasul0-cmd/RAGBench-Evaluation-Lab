"""Bengali text normalization and tokenization utilities for RAGBench.

Design contract (Amendment 1):
  ``normalize_bengali_text`` and ``get_normalization_view`` produce a PROCESSING VIEW
  of the text for BM25 tokenization and embedding input. They NEVER mutate the
  authoritative raw document text from which ``start_char``/``end_char`` offsets
  are computed and stored.

  The invariant is: for any raw document and stored passage span, after calling
  ``get_normalization_view(raw_text)``, the expression
  ``raw_text[start_char:end_char]`` still yields the correct ground-truth span.
  Normalization coordinates (if needed for search) must be maintained in a
  separate coordinate space and never used to rewrite stored offsets.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Bengali stopwords (processing-view only — not used in offset calculation)
# ---------------------------------------------------------------------------
BENGALI_STOPWORDS: frozenset[str] = frozenset(
    {
        # Common function words / particles
        "\u098f",
        "\u098f\u0995\u099f\u09bf",
        "\u098f\u0995\u099f\u09be",
        "\u098f\u0995",
        "\u09a5\u09c7\u0995\u09c7",
        "\u098f\u09ac\u0982",
        "\u0995\u09bf\u09a8\u09cd\u09a4\u09c1",
        "\u0986\u09b0",
        "\u0993",
        "\u09a4\u09ac\u09c7",
        "\u09af\u09a6\u09bf",
        "\u09af\u09c7",
        "\u09af\u09be",
        "\u09af\u09be\u09df",
        "\u09af\u09be\u09ac\u09c7",
        "\u09b9\u09df",
        "\u09b9\u09df\u09c7",
        "\u09b9\u09df\u09c7\u099b\u09c7",
        "\u09b9\u09ac\u09c7",
        "\u09b9\u09bf\u09b8\u09c7\u09ac\u09c7",
        "\u09b9\u09b2",
        "\u0995\u09b0\u09be",
        "\u0995\u09b0\u09c7",
        "\u0995\u09b0\u09c7\u09a8",
        "\u0995\u09b0\u09be\u09b0",
        "\u09a6\u09bf\u09df\u09c7",
        "\u09a6\u09bf\u09df\u09c7\u099b\u09c7",
        "\u09a6\u09bf\u09af\u09bc\u09c7",
        "\u09a6\u09bf\u09df\u09be",
        "\u09a8\u09bf\u09df\u09c7",
        "\u09a8\u09bf\u09df\u09c7\u099b\u09c7",
        "\u09a8\u09c7\u0993\u09df\u09be",
        "\u09a8\u09be",
        "\u09a8\u09df",
        "\u09a8\u09bf",
        "\u09a8\u09bf\u09b2",
        "\u09a8\u09be\u09b9\u09b2\u09c7",
        "\u099b\u09bf\u09b2",
        "\u099b\u09bf\u09b2\u09c7\u09a8",
        "\u099b\u09c7",
        "\u09b8\u09c7",
        "\u09b8\u09c7\u0987",
        "\u09ac\u09b2\u09be",
        "\u09ac\u09b2\u09c7",
        "\u09ac\u09b2\u09c7\u09a8",
        "\u09ac\u09be",
        "\u09ac\u09be\u09dc\u09be",
        "\u09b8\u09be\u09a5\u09c7",
        "\u09b8\u09ae\u09cd\u09aa\u09b0\u09cd\u0995\u09c7",
        "\u09ac\u09bf\u09b7\u09df\u09c7",
        "\u09b8\u09ac",
        "\u09b8\u09b0\u09cd\u09ac",
        "\u09b8\u09b0\u09cd\u09ac\u09be\u0987",
        "\u09aa\u09cd\u09b0\u09a4\u09bf",
        "\u09a4\u09be\u09b0",
        "\u09a4\u09be\u09b0\u09be",
        "\u09a4\u09be\u09b0\u09be\u09b0",
        "\u09a4\u09be\u09a6\u09c7\u09b0",
        "\u0986\u09ae\u09be\u09b0",
        "\u0986\u09ae\u09be\u09a6\u09c7\u09b0",
        "\u0986\u09ae\u09bf",
        "\u0986\u09ae\u09b0\u09be",
        "\u0986\u09aa\u09a8\u09be\u09b0",
        "\u0986\u09aa\u09a8\u09bf",
        "\u0986\u09aa\u09a8\u09be\u09a6\u09c7\u09b0",
        "\u0995\u09c7",
        "\u0995\u09be\u09b0",
        "\u0995\u09be\u09b0\u09a3",
        "\u0995\u09bf",
        "\u0995\u09c0",
        "\u09af\u09c7\u09a4\u09c7",
        "\u09a6\u09bf\u0995\u09c7",
        "\u09ae\u09a7\u09cd\u09af\u09c7",
        "\u099c\u09a8\u09cd\u09af",
        "\u09b9\u09cb\u0995",
        "\u09a5\u09be\u0995\u09c7",
        "\u09a5\u09be\u0995\u09be",
        "\u09a5\u09be\u0995\u09be\u09b0",
        "\u09a4\u09be\u0987",
        "\u09a4\u09be\u09b9\u09b2\u09c7",
        "\u09aa\u09b0\u09c7",
        "\u0986\u0997\u09c7",
        "\u09aa\u09b0",
        "\u09aa\u09a6\u09cd\u09a7\u09a4\u09bf",
        # Determiners / demonstratives
        "\u098f\u0987",
        "\u0993\u0987",
        "\u0995\u09cb\u09a8\u09cb",
        "\u09af\u09c7\u0995\u09cb\u09a8\u09cb",
        # Numbers (Bengali script)
        "\u09e6",
        "\u09e7",
        "\u09e8",
        "\u09e9",
        "\u09ea",
        "\u09eb",
        "\u09ec",
        "\u09ed",
        "\u09ee",
        "\u09ef",
    }
)

# Regex: sentence terminators — Western (.!?) and Bengali/Indic (।\u0964 and ॥\u0965)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?\u0964\u0965])\s+")

# Token pattern: non-whitespace, non-punctuation runs (Unicode-aware)
_PUNCTUATION_CHARS = (
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "\u0964\u0965"  # Dari / Double Dari
    "\u09f7\u09f8"  # Bengali currency numerators
)
_TOKEN_RE = re.compile(rf"[^\s{re.escape(_PUNCTUATION_CHARS)}]+")


def get_normalization_view(raw_text: str) -> str:
    """Return a NFKC-normalized processing view of *raw_text* for BM25/embedding input.

    This function is a processing view only.  It never mutates or reinterprets
    the authoritative raw-text coordinate system.  The caller's stored
    ``start_char``/``end_char`` offsets remain valid indices into the original
    *raw_text* even after calling this function.
    """
    # Preserve Zero-Width Non-Joiner (ZWNJ) and Zero-Width Joiner (ZWJ)
    # by tagging them before NFKC (which may absorb them) then restoring.
    # NFKC can change text length; offsets into raw_text are unaffected.
    tagged = raw_text.replace("\u200c", "\x00ZWNJ\x00").replace("\u200d", "\x00ZWJ\x00")
    normalized = unicodedata.normalize("NFKC", tagged)
    return normalized.replace("\x00ZWNJ\x00", "\u200c").replace("\x00ZWJ\x00", "\u200d")


def normalize_bengali_text(text: str) -> str:
    """Alias for :func:`get_normalization_view` — returns NFKC processing view.

    The name ``normalize_bengali_text`` is kept for callers that do not need to
    emphasise the processing-view distinction.  The behaviour and contract are
    identical: raw document offsets are never modified.
    """
    return get_normalization_view(text)


def tokenize_bengali_text(
    text: str,
    remove_stopwords: bool = False,
) -> list[str]:
    """Tokenize Bengali/English mixed text for BM25 indexing.

    Applies NFKC normalization (processing view), lower-cases, extracts
    Unicode token runs, and optionally removes high-frequency Bengali
    stopwords.

    Args:
        text: Raw input text (Bengali, English, or mixed).
        remove_stopwords: If True, filter tokens against BENGALI_STOPWORDS.

    Returns:
        List of lowercase token strings.
    """
    view = get_normalization_view(text).lower()
    tokens = _TOKEN_RE.findall(view)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in BENGALI_STOPWORDS]
    return tokens


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences at Western (.!?) and Bengali (।॥) boundaries.

    Splitting operates on the raw text to preserve original character positions.
    Returns a list of sentence strings; each sentence retains leading/trailing
    whitespace trimmed.
    """
    parts = _SENTENCE_END_RE.split(text)
    return [p.strip() for p in parts if p.strip()]
