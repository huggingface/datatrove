import datetime
import heapq
import itertools
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import IO, Callable, TextIO

import humanize


INDENT = " " * 4


class MetricStatsDict(defaultdict):
    """
    Stores multiple stats
    """

    def __init__(self, *_, init=None, **kwargs):
        super().__init__(MetricStats, **kwargs)
        if init:
            self.update(init)

    def __add__(self, other):
        result = MetricStatsDict()
        for key, item in itertools.chain(self.items(), other.items()):
            result[key] += item
        return result

    def topk(self, k=20):
        """

        Args:
          k:  (Default value = 20)

        Returns:

        """
        return MetricStatsDict(init={s: self.get(s) for s in heapq.nlargest(k, self, key=lambda s: self.get(s).total)})

    def __repr__(self):
        return ", ".join(f"{key}: {stats}" for key, stats in self.items())

    def to_dict(self):
        return {a: (b.to_dict() if hasattr(b, "to_dict") else b) for a, b in self.items()}


class Stats:
    """
    Stats for a particular block

    Args:
        name: The name of the block
    """

    def __init__(self, name: str):
        self.name = name
        self.time_stats = TimingStats()
        self.stats = MetricStatsDict()

    def __getitem__(self, item: str) -> "MetricStats":
        return self.stats[item]

    def __setitem__(self, key: str, value: "MetricStats"):
        self.stats[key] = value

    def __add__(self, stat):
        assert self.name == stat.name, f"Can not merge stats from different blocks {self.name} != {stat.name}"
        result = Stats(self.name)
        result.time_stats = self.time_stats + stat.time_stats
        result.stats = self.stats + stat.stats
        return result

    def __repr__(self, total_time: float = 0.0):
        return f"\n{INDENT}".join(
            filter(
                lambda x: x is not None,
                [
                    f"{self.name}",
                    f"Runtime: {self.time_stats.get_repr(total_time)}" if self.time_stats.total > 0 else None,
                    f"Stats: {{{self.stats}}}" if len(self.stats) > 0 else None,
                ],
            )
        )

    def to_dict(self):
        return {
            "name": self.name,
            "time_stats": self.time_stats.to_dict(),
            "stats": self.stats.to_dict(),
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    def save_to_disk(self, file: TextIO):
        """

        Args:
          file: TextIO:

        Returns:

        """
        file.write(self.to_json())

    @classmethod
    def from_dict(cls, data):
        """

        Args:
          data:

        Returns:

        """
        stats = cls(data["name"])
        stats.time_stats = TimingStats.from_dict(data["time_stats"])
        stats.stats = MetricStatsDict(init=data["stats"])
        if doc_len_stats := data.get("doc_len_stats", None):  # backwards compatibility
            stats.stats["doc_len"] = MetricStats.from_dict(doc_len_stats)
        return stats


class PipelineStats:
    def __init__(self, stats: list[Stats | Callable] = None):
        self.stats: list[Stats] = stats if stats else []
        if self.stats and not isinstance(self.stats[0], Stats):
            self.stats: list[Stats] = [
                pipeline_step.stats for pipeline_step in self.stats if hasattr(pipeline_step, "stats")
            ]

    def __add__(self, pipestat):
        if not self.stats:
            return PipelineStats(pipestat.stats)
        return PipelineStats([x + y for x, y in zip(self.stats, pipestat.stats)])

    @property
    def total_time(self):
        return sum([stat.time_stats.global_mean for stat in self.stats])

    @property
    def total_std_dev(self):
        return math.sqrt(sum((stat.time_stats.global_std_dev**2 for stat in self.stats)))

    def get_repr(self, text=None):
        """

        Args:
          text:  (Default value = None)

        Returns:

        """
        total_time = self.total_time
        total_std_dev = self.total_std_dev
        x = (
            f"\n\n{'📉' * 3} Stats{': ' + text if text else ''} {'📉' * 3}\n\n"
            + f"Total Runtime: {humanize.precisedelta(total_time)}"
            + (f" ± {humanize.precisedelta(total_std_dev)}/task" if total_std_dev != 0.0 else "")
            + "\n\n"
        )
        x += "\n".join([stat.__repr__(total_time) for stat in self.stats])
        return x

    def __repr__(self):
        return self.get_repr()

    def to_json(self):
        return json.dumps([stat.to_dict() for stat in self.stats], indent=4)

    @classmethod
    def from_json(cls, data):
        """

        Args:
          data:

        Returns:

        """
        return PipelineStats([Stats.from_dict(stat) for stat in data])

    def save_to_disk(self, file: IO):
        """

        Args:
          file: IO:

        Returns:

        """
        file.write(self.to_json())


@dataclass
class MetricStats:
    """
    Stats to track a particular metric
    """

    total: float = 0
    n: int = 0
    mean: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    _running_variance: float = 0.0
    unit: str = "doc"

    def update(self, x: float, unit: str = None):
        """

        Args:
          x: float:
          unit: str:  (Default value = None)

        Returns:

        """
        if unit:
            self.unit = unit
        self.total += x
        self.n += 1

        self.min = min(self.min, x)
        self.max = max(self.max, x)

        # https://en.wikipedia.org/wiki/Algorithms_.for_calculating_variance#Welford%27s_online_algorithm
        delta = x - self.mean
        self.mean += delta / self.n
        if self.n > 1:
            self._running_variance += delta * (x - self.mean)

    @property
    def variance(self):
        return self._running_variance / (self.n - 1) if self.n > 1 else 0.0

    @property
    def standard_deviation(self):
        return math.sqrt(self.variance)

    def __add__(self, other):
        if not isinstance(other, type(self)):
            other = type(self).from_dict(other)
        # mean and variance: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        n = self.n + other.n
        mean = 0.0
        _running_variance = 0.0
        if self.n + other.n > 0:
            mean = (self.n * self.mean + other.n * other.mean) / n
            delta = self.mean - other.mean
            _running_variance = (
                self._running_variance + other._running_variance + (delta * delta * self.n * other.n) / n
            )

        total = self.total + other.total
        M = max(self.max, other.max)
        m = min(self.min, other.min)
        return type(self)(
            total=total,
            n=n,
            mean=mean,
            min=m,
            max=M,
            _running_variance=_running_variance,
            unit=self.unit if self.unit != "doc" else other.unit,
        )

    def to_dict(self):
        if self.total == 0:
            return 0
        data = {
            "total": self.total,
        }
        # only relevant if > 1 and we didn't just add 1 all the time
        if self.n > 1 and self.n != self.total:
            data["n"] = self.n
        if self.mean != 1:
            data["mean"] = self.mean
        # are there actually different values
        if self.mean != self.max or self.mean != self.min:
            data["mean"] = self.mean
            data["variance"] = self.variance
            data["std_dev"] = self.standard_deviation
            data["min"] = self.min
            data["max"] = self.max
        if self.unit != "doc":
            data["unit"] = self.unit
        return self.total if len(data) == 1 else data

    @classmethod
    def from_dict(cls, data):
        """

        Args:
          data:

        Returns:

        """
        if isinstance(data, dict):
            total = data.get("total")
            mean = data.get("mean", 1)
            n = data.get("n", total if mean != 1 else 1)
            return cls(
                total=total,
                n=n,
                mean=mean,
                min=data.get("min", mean),
                max=data.get("max", mean),
                _running_variance=data.get("variance", 0.0) * (n - 1),
                unit=data.get("unit", cls.unit),
            )
        else:
            return cls(total=data, min=data, max=data, mean=data, n=1, unit="task")

    def __repr__(self):
        if self.mean != 1:
            # only display relevant fields
            elements = [
                (f"min={self.min}, ", self.min != self.mean),
                (f"max={self.max}, ", self.max != self.mean),
                (f"{self.mean:.2f}", True),
                (f"±{self.standard_deviation:.0f}", self.standard_deviation != 0.0),
            ]
            return f"{self.total} [" + "".join([t for t, c in elements if c]) + f"/{self.unit}]"
        return str(self.total)


@dataclass
class TimingStats(MetricStats):
    global_mean: float = 0
    n_tasks: int = 1
    global_min: float = float("inf")
    global_max: float = float("-inf")
    global_std_dev: float = 0.0

    def __enter__(self):
        self._entry_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.update(time.perf_counter() - self._entry_time)

    def __post_init__(self):
        if self.global_mean == 0:
            self.global_mean = self.global_min = self.global_max = self.total

    def update(self, x: float, unit: str = None):
        """

        Args:
          x: float:
          unit: str:  (Default value = None)

        Returns:

        """
        super().update(x, unit)
        if self.n_tasks == 1:
            self.global_mean = self.global_min = self.global_max = self.total

    def __add__(self, other):
        new_time_stats: TimingStats = super().__add__(other)
        if not isinstance(other, type(self)):
            other = type(self).from_dict(other)
        new_time_stats.global_min = min(self.global_min, other.global_min)
        new_time_stats.global_max = max(self.global_max, other.global_max)
        new_time_stats.n_tasks = self.n_tasks + other.n_tasks
        new_time_stats.global_mean = (
            self.n_tasks * self.global_mean + other.n_tasks * other.global_mean
        ) / new_time_stats.n_tasks
        s1 = self.global_std_dev**2 * (self.n_tasks - 1)
        s2 = other.global_std_dev**2 * (other.n_tasks - 1)
        s = (
            s1
            + s2
            + (self.global_mean - other.global_mean) ** 2 * self.n_tasks * other.n_tasks / new_time_stats.n_tasks
        )
        new_time_stats.global_std_dev = math.sqrt(s / (new_time_stats.n_tasks - 1))
        return new_time_stats

    def _get_time_frac(self, total_time):
        """

        Args:
          total_time:

        Returns:

        """
        return (self.global_mean / total_time) if total_time > 0 else 0

    def get_repr(self, total_time: float = 0.0):
        """

        Args:
          total_time: float:  (Default value = 0.0)

        Returns:

        """
        if self.total == 0:
            return "Time not computed"
        return (
            f"({self._get_time_frac(total_time):.2%})"
            + f" {humanize.precisedelta(self.global_mean)}"
            + (f"±{humanize.precisedelta(self.global_std_dev)}/task" if self.global_std_dev != 0 else "")
            + (f", min={humanize.precisedelta(self.global_min)}" if self.global_min != self.total else "")
            + (f", max={humanize.precisedelta(self.global_max)}" if self.global_max != self.total else "")
            + f" [{humanize.precisedelta(datetime.timedelta(seconds=self.mean), minimum_unit='milliseconds')}"
            + f"±{humanize.precisedelta(datetime.timedelta(seconds=self.standard_deviation), minimum_unit='milliseconds')}/{self.unit}]"
        )

    def __repr__(self):
        return self.get_repr()

    def to_dict(self):
        data = super().to_dict()
        if isinstance(data, dict):
            data["total_human"] = humanize.precisedelta(self.total)
            data["mean_human"] = humanize.precisedelta(
                datetime.timedelta(seconds=self.mean), minimum_unit="milliseconds"
            )
            data["std_dev_human"] = humanize.precisedelta(
                datetime.timedelta(seconds=self.standard_deviation), minimum_unit="milliseconds"
            )
            data["min_human"] = humanize.precisedelta(
                datetime.timedelta(seconds=self.min), minimum_unit="milliseconds"
            )
            data["max_human"] = humanize.precisedelta(
                datetime.timedelta(seconds=self.max), minimum_unit="milliseconds"
            )
            if self.n_tasks != 1:
                data["global_mean"] = self.global_mean
                data["global_mean_human"] = humanize.precisedelta(self.global_mean)
                data["global_min"] = self.global_min
                data["global_min_human"] = humanize.precisedelta(self.global_min)
                data["global_max"] = self.global_max
                data["global_max_human"] = humanize.precisedelta(self.global_max)
                data["global_std_dev"] = self.global_std_dev
                data["global_std_dev_human"] = humanize.precisedelta(self.global_std_dev)
        return data

    @classmethod
    def from_dict(cls, data):
        """

        Args:
          data:

        Returns:

        """
        res: TimingStats = super().from_dict(data)
        if isinstance(data, dict):
            res.global_mean = data.get("global_mean", res.total)
            res.global_min = data.get("global_min", res.total)
            res.global_max = data.get("global_max", res.total)
            res.global_std_dev = data.get("global_std_dev", 0)
        else:
            res.global_mean = res.global_min = res.global_max = res.total
        return res
