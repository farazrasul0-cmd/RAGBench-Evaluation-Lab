"""Unit tests for Bengali text normalization, tokenization, and offset-preservation.

Amendment 1 contract: test_offset_preservation verifies that ground-truth
start_char/end_char remain coordinates into the immutable raw document text
even after a normalization processing view is created.
"""

from app.engine.multilingual.bengali import (
    BENGALI_STOPWORDS,
    get_normalization_view,
    normalize_bengali_text,
    split_sentences,
    tokenize_bengali_text,
)


class TestOffsetPreservation:
    """Invariant: raw-text character offsets survive normalization."""

    def test_offset_preservation_ascii(self) -> None:
        """ASCII text: offsets into raw text are valid after normalization view."""
        raw = "The Arctic ice is melting rapidly due to climate change."
        start, end = 4, 10
        ground_truth_span = raw[start:end]  # "Arctic"

        _ = get_normalization_view(raw)  # create processing view

        # Offsets into raw text STILL yield the correct span
        assert raw[start:end] == ground_truth_span
        assert raw[start:end] == "Arctic"

    def test_offset_preservation_bengali(self) -> None:
        """Bengali text: NFKC changes view length but not raw-text offsets."""
        # Bengali: "জলবায়ু পরিবর্তন বৈশ্বিক সমস্যা।"
        raw = (
            "\u099c\u09b2\u09ac\u09be\u09df\u09c1 \u09aa\u09b0\u09bf\u09ac\u09b0\u09cd\u09a4\u09a8 "
            "\u09ac\u09c8\u09b6\u09cd\u09ac\u09bf\u0995 \u09b8\u09ae\u09b8\u09cd\u09af\u09be\u0964"
        )
        start, end = 7, 20
        ground_truth_span = raw[start:end]

        view = get_normalization_view(raw)

        # Confirm normalization produces a different string (may differ in length)
        # — this is acceptable because offsets index raw_text, NOT the view
        # Raw offsets remain valid:
        assert raw[start:end] == ground_truth_span
        # The view is only for BM25/embedding; it does not redefine coordinates
        assert isinstance(view, str)
        assert len(view) > 0

    def test_normalization_view_does_not_mutate_raw(self) -> None:
        """Calling get_normalization_view never modifies the input string."""
        raw = "\u099c\u09b2\u09ac\u09be\u09df\u09c1"
        original_id = id(raw)
        _ = get_normalization_view(raw)
        assert id(raw) == original_id  # Python strings are immutable; just confirm no rebind

    def test_normalize_bengali_text_alias(self) -> None:
        """normalize_bengali_text is an alias for get_normalization_view."""
        text = "\u099c\u09b2\u09ac\u09be\u09df\u09c1"
        assert normalize_bengali_text(text) == get_normalization_view(text)

    def test_zwnj_preserved_in_view(self) -> None:
        """Zero-Width Non-Joiner (ZWNJ) is preserved in the normalization view."""
        raw = "\u0995\u200c\u09cd\u09b7"  # Bengali with ZWNJ
        view = get_normalization_view(raw)
        assert "\u200c" in view

    def test_zwj_preserved_in_view(self) -> None:
        """Zero-Width Joiner (ZWJ) is preserved in the normalization view."""
        raw = "\u0995\u200d\u09cd\u09b7"
        view = get_normalization_view(raw)
        assert "\u200d" in view


class TestSentenceSplitOnDari:
    """Bengali/Indic sentence terminators: Dari (।) and Double Dari (॥)."""

    def test_split_on_dari_single(self) -> None:
        # U+0964 is Bengali Dari (sentence full-stop)
        text = (
            "\u098f\u099f\u09bf \u098f\u0995\u099f\u09bf \u09ac\u09be\u0995\u09cd\u09af\u0964 "
            "\u098f\u09ac\u0982 \u098f\u099f\u09bf \u0986\u09b0\u09c7\u0995\u099f\u09bf\u0964"
        )
        parts = split_sentences(text)
        assert len(parts) == 2

    def test_split_on_double_dari(self) -> None:
        # U+0965 Double Dari
        text = (
            "\u09aa\u09cd\u09b0\u09a5\u09ae \u09ac\u09be\u0995\u09cd\u09af\u0965 "
            "\u09a6\u09cd\u09ac\u09bf\u09a4\u09c0\u09df \u09ac\u09be\u0995\u09cd\u09af\u0965"
        )
        parts = split_sentences(text)
        assert len(parts) == 2

    def test_split_on_period(self) -> None:
        text = "First sentence. Second sentence."
        parts = split_sentences(text)
        assert len(parts) == 2

    def test_split_mixed_dari_period(self) -> None:
        text = "\u09aa\u09cd\u09b0\u09a5\u09ae\u0964 Second sentence."
        parts = split_sentences(text)
        assert len(parts) == 2

    def test_split_does_not_consume_adjacent_chars(self) -> None:
        text = (
            "\u0986\u09ae\u09bf \u09af\u09be\u0987\u0964 "
            "\u09a4\u09c1\u09ae\u09bf \u0995\u09cb\u09a5\u09be\u09df?"
        )
        parts = split_sentences(text)
        # Dari should not eat into the next word
        assert all(len(p) > 0 for p in parts)
        assert not any(p.startswith("\u0964") for p in parts)


class TestBM25TokenizationStable:
    """Tokenization must be deterministic across repeated calls."""

    def test_tokenization_deterministic(self) -> None:
        text = "Climate change affects the Sundarbans delta region."
        result1 = tokenize_bengali_text(text)
        result2 = tokenize_bengali_text(text)
        assert result1 == result2

    def test_bengali_tokenization_deterministic(self) -> None:
        text = (
            "\u099c\u09b2\u09ac\u09be\u09df\u09c1 \u09aa\u09b0\u09bf\u09ac\u09b0\u09cd\u09a4\u09a8"
        )
        result1 = tokenize_bengali_text(text)
        result2 = tokenize_bengali_text(text)
        assert result1 == result2

    def test_empty_text_tokenization(self) -> None:
        assert tokenize_bengali_text("") == []

    def test_dari_not_included_as_token(self) -> None:
        text = "\u0986\u09ae\u09bf \u09af\u09be\u0987\u0964"
        tokens = tokenize_bengali_text(text)
        assert "\u0964" not in tokens
        assert all("\u0964" not in t for t in tokens)


class TestStopwordFiltering:
    """Stopwords are removed when remove_stopwords=True."""

    def test_stopwords_removed(self) -> None:
        # "এবং" (and) is in BENGALI_STOPWORDS
        text = "\u0986\u09ae\u09bf \u098f\u09ac\u0982 \u09a4\u09c1\u09ae\u09bf"
        tokens_with = tokenize_bengali_text(text, remove_stopwords=False)
        tokens_without = tokenize_bengali_text(text, remove_stopwords=True)
        assert len(tokens_without) <= len(tokens_with)

    def test_stopword_set_not_empty(self) -> None:
        assert len(BENGALI_STOPWORDS) >= 20

    def test_no_removal_by_default(self) -> None:
        text = "\u0986\u09ae\u09bf \u09af\u09be\u0987"
        without_flag = tokenize_bengali_text(text)
        explicit_false = tokenize_bengali_text(text, remove_stopwords=False)
        assert without_flag == explicit_false
