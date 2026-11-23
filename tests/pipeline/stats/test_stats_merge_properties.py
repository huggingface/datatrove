import json
import math

import numpy as np
import pytest


pytest.importorskip("hypothesis")
from hypothesis import given
from hypothesis import strategies as st

from datatrove.utils.stats import MetricStats, PipelineStats, Stats, TimingStats


# Per-rank stats are saved with to_dict() and merged after from_dict() (StatsMerger, merge_stats, Jobs, Ray).
# Oracle: merging the per-rank stats must match numpy on all the values.


def build_metric_stats(values, cls=MetricStats):
    stats = cls()
    for value in values:
        stats.update(value)
    return stats


def save_and_load(stats, cls=MetricStats):
    return cls.from_dict(json.loads(json.dumps(stats.to_dict())))


def assert_matches_numpy(stats, values):
    array = np.asarray(values, dtype=float)
    expected_variance = float(np.var(array, ddof=1)) if len(array) > 1 else 0.0
    assert stats.n == len(values)
    assert math.isclose(stats.total, array.sum(), rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(stats.mean, array.mean(), rel_tol=1e-9, abs_tol=1e-9)
    assert (stats.min, stats.max) == (array.min(), array.max())
    assert math.isclose(stats.variance, expected_variance, rel_tol=1e-6, abs_tol=1e-6)


def merge_saved(parts):
    merged = MetricStats()
    for part in parts:
        merged += save_and_load(build_metric_stats(part))
    return merged


def all_values(parts):
    return [value for part in parts for value in part]


# small values make edge cases likely: zero totals, all-ones counters, values that cancel out
values = st.lists(st.integers(-3, 3) | st.integers(-1000, 1000), min_size=1, max_size=20)


# ranks can be empty (a rank with no documents), as long as some rank has values
@given(parts=st.lists(values | st.just([]), min_size=1, max_size=5).filter(all_values))
def test_merge_in_memory_matches_numpy(parts):
    merged = sum((build_metric_stats(part) for part in parts), start=MetricStats())
    assert_matches_numpy(merged, all_values(parts))


# Excluded until the known bugs below are fixed: ranks with a total of 0, and ranks where total == n (n is then
# not saved, and is rebuilt only if the mean is exactly 1). This also excludes all-ones ranks, which are saved
# as a single per-task count by design.
saved_part = values.filter(lambda part: sum(part) != 0 and sum(part) != len(part))


@given(parts=st.lists(saved_part, min_size=1, max_size=5))
def test_merge_of_saved_stats_matches_numpy(parts):
    assert_matches_numpy(merge_saved(parts), all_values(parts))


# Known bugs. strict=True: once fixed these pass and fail CI until the markers are removed.
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="to_dict() saves a zero total as 0, loaded as n=1")
@pytest.mark.parametrize("parts", [[[0, 0]], [[0, 0], [2, 2]], [[-1, 1]]])
def test_merge_of_saved_stats_with_zero_total(parts):
    assert_matches_numpy(merge_saved(parts), all_values(parts))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="n is not saved when n == total; mean is not exactly 1")
# in both cases the running mean is 0.9999999999999999
@pytest.mark.parametrize("part", [[0.1, 1.9], [0] * 10 + [12, 0]])
def test_saved_stats_keep_n_when_total_equals_n(part):
    assert save_and_load(build_metric_stats(part)).n == len(part)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="stats + empty PipelineStats drops all stats")
def test_adding_empty_pipeline_stats_on_the_right_keeps_stats():
    step_stats = Stats("step")
    step_stats["metric"].update(5)
    merged = PipelineStats([step_stats]) + PipelineStats()
    assert [stats.name for stats in merged.stats] == ["step"]
    assert merged.stats[0]["metric"].total == 5


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="TimingStats.to_dict() does not save n_tasks")
def test_merging_saved_merged_timing_stats_counts_all_tasks():
    tasks = [build_metric_stats([seconds], TimingStats) for seconds in (1.0, 2.0, 3.0)]
    first_two = save_and_load(tasks[0] + tasks[1], TimingStats)
    assert (first_two + tasks[2]).n_tasks == 3
