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
        self.total, self.n = 0, 0
        self._running_mean, self._running_variance = 0, 0
        self.max_value, self.min_value = 0, float('inf')

    def update(self, x: float):
        self.n += 1
        self.total += x

        self.min_value = min(self.min_value, x)
        self.max_value = max(self.max_value, x)
        previous_running_mean = self._running_mean
        self._running_mean = previous_running_mean + (x - previous_running_mean) / self.n
        if self.n == 1:
            self._running_variance = 0
        else:
            self._running_variance = self._running_variance + (x - previous_running_mean) * (x - self._running_mean) / (
                        self.n - 1)

    def get_stats(self) -> dict:
        return {"mean": self._running_mean,
                "variance": self._running_variance,
                "standard_deviation": math.sqrt(self._running_variance),
                "total_time": self.total,
                "count": self.n,
                "max": self.max_value,
                "min": self.min_value,
                }


class TimeStatsManager(RunningStats):
    def __init__(self):
        super().__init__()

    def __enter__(self):
        self._entry_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._entry_time is not None
        self.update(time.perf_counter() - self._entry_time)


def merge_time_stats_couple(stat_1: dict, stat_2: dict) -> dict:
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
