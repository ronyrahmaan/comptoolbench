"""Tests for argument matching functions."""

from __future__ import annotations

from comptoolbench.evaluation.matchers import (
    levenshtein_distance,
    match_arguments,
    match_exact,
    match_numeric,
    match_string,
    match_value,
    normalized_similarity,
)


class TestLevenshteinDistance:
    def test_identical(self) -> None:
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self) -> None:
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3

    def test_single_edit(self) -> None:
        assert levenshtein_distance("cat", "hat") == 1  # substitution
        assert levenshtein_distance("cat", "cats") == 1  # insertion
        assert levenshtein_distance("cats", "cat") == 1  # deletion

    def test_completely_different(self) -> None:
        assert levenshtein_distance("abc", "xyz") == 3


class TestNormalizedSimilarity:
    def test_identical(self) -> None:
        assert normalized_similarity("hello", "hello") == 1.0

    def test_empty(self) -> None:
        assert normalized_similarity("", "") == 1.0

    def test_completely_different(self) -> None:
        assert normalized_similarity("abc", "xyz") == 0.0

    def test_partial(self) -> None:
        sim = normalized_similarity("hello", "hallo")
        assert 0.7 < sim < 1.0  # One char diff out of 5


class TestMatchExact:
    def test_identical(self) -> None:
        assert match_exact("get_weather", "get_weather") is True

    def test_case_insensitive(self) -> None:
        assert match_exact("Get_Weather", "get_weather") is True

    def test_whitespace(self) -> None:
        assert match_exact("  hello ", "hello") is True

    def test_different(self) -> None:
        assert match_exact("get_weather", "get_time") is False

    def test_numbers(self) -> None:
        assert match_exact(42, "42") is True

    def test_booleans(self) -> None:
        assert match_exact(True, "True") is True


class TestMatchString:
    def test_exact(self) -> None:
        assert match_string("hello world", "hello world") == 1.0

    def test_case_whitespace_normalized(self) -> None:
        assert match_string("Hello  World", "hello world") == 1.0

    def test_similar_above_threshold(self) -> None:
        score = match_string("San Francisco", "San Franciscoo")
        assert score > 0.85

    def test_different_below_threshold(self) -> None:
        score = match_string("New York", "San Francisco")
        assert score == 0.0

    def test_custom_threshold(self) -> None:
        # "cat" vs "hat" is ~0.67 similar
        assert match_string("cat", "hat", threshold=0.5) > 0
        assert match_string("cat", "hat", threshold=0.8) == 0.0


class TestMatchNumeric:
    def test_exact(self) -> None:
        assert match_numeric(42.0, 42.0) is True

    def test_close(self) -> None:
        assert match_numeric(100.0, 100.001, rel_tol=0.01) is True

    def test_not_close(self) -> None:
        assert match_numeric(100.0, 200.0) is False

    def test_zero_expected(self) -> None:
        assert match_numeric(0.0, 0.00001) is True
        assert match_numeric(0.0, 1.0) is False

    def test_integers(self) -> None:
        assert match_numeric(42.0, 42.0) is True


class TestMatchValue:
    def test_none_both(self) -> None:
        assert match_value(None, None) == 1.0

    def test_none_one(self) -> None:
        assert match_value(None, "hello") == 0.0
        assert match_value("hello", None) == 0.0

    def test_booleans(self) -> None:
        assert match_value(True, True) == 1.0
        assert match_value(True, False) == 0.0

    def test_numbers(self) -> None:
        assert match_value(42, 42) == 1.0
        assert match_value(42, 43) == 0.0  # Outside rel_tol
        assert match_value(42.0, "42") == 1.0  # String that parses to number

    def test_strings(self) -> None:
        assert match_value("hello", "hello") == 1.0
        assert match_value("hello", "world") == 0.0

    def test_lists_identical(self) -> None:
        assert match_value([1, 2, 3], [1, 2, 3]) == 1.0

    def test_lists_different_length(self) -> None:
        score = match_value([1, 2], [1, 2, 3])
        assert 0 < score < 1.0  # Penalized for length mismatch

    def test_lists_empty(self) -> None:
        assert match_value([], []) == 1.0
        assert match_value([], [1]) == 0.0

    def test_dicts_identical(self) -> None:
        assert match_value({"a": 1, "b": 2}, {"a": 1, "b": 2}) == 1.0

    def test_dicts_missing_key(self) -> None:
        score = match_value({"a": 1, "b": 2}, {"a": 1})
        assert score == 0.5  # One key matches, one missing

    def test_dicts_extra_key(self) -> None:
        score = match_value({"a": 1}, {"a": 1, "b": 2})
        assert score == 0.5  # Extra key penalized

    def test_dicts_empty(self) -> None:
        assert match_value({}, {}) == 1.0

    def test_type_mismatch_list(self) -> None:
        assert match_value([1, 2], "not a list") == 0.0

    def test_type_mismatch_dict(self) -> None:
        assert match_value({"a": 1}, "not a dict") == 0.0


class TestMatchArguments:
    def test_both_empty(self) -> None:
        score, per_arg = match_arguments({}, {})
        assert score == 1.0
        assert per_arg == {}

    def test_one_empty(self) -> None:
        score, _ = match_arguments({"a": 1}, {})
        assert score == 0.0

    def test_perfect_match(self) -> None:
        expected = {"city": "San Francisco", "units": "celsius"}
        actual = {"city": "San Francisco", "units": "celsius"}
        score, per_arg = match_arguments(expected, actual)
        assert score == 1.0
        assert per_arg["city"] == 1.0
        assert per_arg["units"] == 1.0

    def test_partial_match(self) -> None:
        expected = {"city": "San Francisco", "units": "celsius"}
        actual = {"city": "San Francisco", "units": "fahrenheit"}
        score, per_arg = match_arguments(expected, actual)
        assert per_arg["city"] == 1.0
        assert per_arg["units"] == 0.0  # Different string below threshold
        assert 0 < score < 1.0

    def test_missing_arg(self) -> None:
        expected = {"city": "NYC", "units": "celsius"}
        actual = {"city": "NYC"}
        score, per_arg = match_arguments(expected, actual)
        assert per_arg["city"] == 1.0
        assert per_arg["units"] == 0.0
        assert score == 0.5

    def test_extra_arg(self) -> None:
        expected = {"city": "NYC"}
        actual = {"city": "NYC", "extra": "value"}
        score, per_arg = match_arguments(expected, actual)
        assert per_arg["city"] == 1.0
        assert per_arg["extra"] == 0.0
        assert score == 0.5
