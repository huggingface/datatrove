"""
    Implements Welford's online algorithm to compute mean and standard deviation of execution time.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm
"""

import math
import time
from collections import Counter

from dataclasses import dataclass


@dataclass
class Stats:
    name: str
    time: dict
    counter: Counter


class RunningStats:
    def __init__(self):
        self.n = 0
        self.total = 0
        self._running_mean = 0
        self._running_variance = 0
        self._time_0 = None

        self.max = None
        self.min = None

    @property
    def mean(self) -> float:
        return self._running_mean if self.n else 0.0

    @property
    def variance(self) -> float:
        return self._running_variance / (self.n - 1) if self.n > 1 else 0.0

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(self.variance)

    def update(self, x: float):
        self.n += 1
        self.total += x

        if self.min is None or x < self.min:
            self.min = x

        if self.max is None or x > self.max:
            self.max = x

        if self.n == 1:
            self._running_mean = x
            self._running_variance = 0
        else:
            previous_rmean = self._running_mean
            self._running_mean = previous_rmean + (x - previous_rmean) / self.n
            self._running_variance = self._running_variance + (x - previous_rmean) * (x - self._running_mean)

    def get_stats(self) -> dict:
        return {"mean": self.mean,
                "variance": self.variance,
                "standard_deviation": self.standard_deviation,
                "total_time": self.total,
                "count": self.n,
                "max": self.max,
                "min": self.min,

                }


class TimeStatsManager(RunningStats):
    def __init__(self):
        super().__init__()

    def __enter__(self):
        self._time_0 = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._time_0 is not None
        x = time.perf_counter() - self._time_0
        self.update(x)


def merge_time_stats_couple(stat_1, stat_2) -> dict:
    n = stat_1["count"] + stat_2["count"]
    merge_mean = (stat_1["count"] * stat_1["mean"] + stat_2["count"] * stat_2["mean"]) / n
    delta = stat_1["mean"] - stat_2["mean"]
    merge_variance = stat_1["variance"] + stat_2["variance"] + stat_1["count"] * stat_2["count"] * delta ** 2 / n
    return {"mean": merge_mean,
            "variance": merge_variance,
            "standard_deviation": math.sqrt(merge_variance),
            "total_time": stat_1["total_time"] + stat_2["total_time"],
            "count": n,
            "max": max([stat_1["max"], stat_2["max"]]),
            "min": min([stat_1["max"], stat_2["max"]]),
            }


def merge_time_stats(stats: list[dict]):
    stats_0 = stats[0]
    for stat in stats[1:]:
        stats_0 = merge_time_stats_couple(stats_0, stat)
    return stats_0


def merge_all(stats: list[list[Stats]]) -> list[(str, Counter)]:
    # first list -> workers, second list -> blocks within pipeline
    final_stats = []
    for i in range(len(stats[0])):
        final_stats.append((stats[0][i].name,
                            merge_time_stats([stats[j][i].time for j in range(len(stats))]),
                            sum([stats[j][i].counter for j in range(len(stats))], Counter())))
    return final_stats
