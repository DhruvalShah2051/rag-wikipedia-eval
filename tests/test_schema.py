"""
Tests for ivfflat index sizing.

These are pure arithmetic, but they guard the setting that produced this
project's most misleading measurement. A fixed `lists = 10` over ~414 chunks
scanned a tenth of the corpus per query, dropped the correct chunk on 2 of 10
benchmark questions, and was recorded as a 90% retrieval score. pgvector raises
no error when an index is sized wrong - it just returns different neighbours.
"""

import pytest

from schema import ROWS_PER_LIST, ivfflat_lists, ivfflat_probes


@pytest.mark.parametrize("rows", [0, 1, 9, 414, 999])
def test_small_corpora_collapse_to_a_single_list(rows):
    """
    Below pgvector's rows-per-list threshold there is one partition, so a scan
    is exhaustive and retrieval is exact. That is the correct behaviour at this
    project's scale, not a degenerate case to work around.
    """
    assert ivfflat_lists(rows) == 1


@pytest.mark.parametrize(
    "rows, expected",
    [(1000, 1), (5000, 5), (50_000, 50), (1_000_000, 1000)],
)
def test_lists_scale_with_the_corpus(rows, expected):
    assert ivfflat_lists(rows) == expected


def test_lists_follow_pgvectors_rows_per_list_guidance():
    assert ivfflat_lists(ROWS_PER_LIST * 7) == 7


def test_lists_is_never_zero():
    """A lists=0 index is invalid DDL and would fail at CREATE INDEX."""
    assert ivfflat_lists(0) >= 1
    assert ivfflat_lists(-5) >= 1


@pytest.mark.parametrize("lists, expected", [(1, 1), (4, 2), (25, 5), (100, 10)])
def test_probes_are_the_square_root_of_lists(lists, expected):
    assert ivfflat_probes(lists) == expected


def test_probes_is_never_zero():
    """probes=0 would scan nothing and return no rows at all."""
    assert ivfflat_probes(0) >= 1


def test_single_list_is_scanned_exhaustively():
    """The property the benchmark's determinism rests on."""
    lists = ivfflat_lists(414)
    assert ivfflat_probes(lists) == lists
