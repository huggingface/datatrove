import math
import time
from collections import Counter
from dataclasses import dataclass


@dataclass
class OnlineStats:
    mean: float
    variance: float
    standard_deviation: float
    total_time: float
    count: int
    max: float
    min: float


class Stats:
    def __init__(self, name: str, indentation: int = 4):
        self.name = name
        self.time_manager = TimeStatsManager()
        self.counter = Counter()
        self.doc_len = ComputeOnlineStats()
        self.indentation = indentation

    @property
    def time(self):
        return self.time_manager.get_stats()

    @property
    def length(self):
        return self.doc_len.get_stats()

    def __add__(self, stat):
        assert self.name == stat.name, f"Summing different blocks {self.name} != {stat.name}"
        self.counter += stat.counter
        self.time_manager += stat.time_manager
        return self

    def _get_frac(self, total_time):
        return (self.time.total_time / total_time) * 100 if total_time > 0 else 0

    def _counter_repr(self):
        if self.counter == Counter():
            return ""
        return f"{' ' * self.indentation + '[' + ''.join([f'{k}={v} ' for k, v in self.counter.items()])}]\n"

    def _time_repr(self, total_time: float):
        if self.time.total_time == 0:
            return f"{' ' * self.indentation}Time not computed\n"
        return (
            f"{' ' * self.indentation}[{self.time.total_time:.4f}s {self._get_frac(total_time):.4f}%"
            f" {self.time.mean:.4f}±{self.time.standard_deviation:.4f}s/doc]\n"
        )

    def _len_repr(self):
        if self.doc_len.counter["n"] == 0:
            return ""
        return (
            f"{' ' * self.indentation}[max={self.length.max}, min={self.length.min}"
            f" {self.length.mean:.0f}±{self.length.standard_deviation:.0f}chars/doc]\n"
        )

    def __repr__(self, total_time: float):
        return f"{self.name}\n" f"{self._time_repr(total_time)}" f"{self._counter_repr()} " f"{self._len_repr()} "


class PipelineStats:
    def __init__(self, stats: list[Stats]):
        self.stats = stats

    def __add__(self, pipestat):
        self.stats = [x + y for x, y in zip(self.stats, pipestat.stats)]
        return self

    @property
    def total_time(self):
        return sum([stat.time.total_time for stat in self.stats])

    def __repr__(self):
        x = f"\n\n{'📉' * 3} STATS {'📉' * 3}\n\n"
        x += "".join([stat.__repr__(self.total_time) for stat in self.stats])
        return x


class ComputeOnlineStats:
    def __init__(self):
        self.counter = Counter()
        self.running_mean, self.running_variance = 0, 0
        self.max_value, self.min_value = 0.0, float("inf")

    def update(self, x: float):
        self.counter["total"] += x
        self.counter["n"] += 1

        self.min_value = min(self.min_value, x)
        self.max_value = max(self.max_value, x)
        previous_running_mean = self.running_mean
        self.running_mean = previous_running_mean + (x - previous_running_mean) / self.counter["n"]
        if self.counter["n"] == 1:
            self.running_variance = 0
        else:
            self.running_variance = self.running_variance + (x - previous_running_mean) * (x - self.running_mean) / (
                self.counter["n"] - 1
            )

    def __add__(self, other):
        self.counter["total"] += other.counter["total"]
        n = self.counter["n"] + other.counter["n"]
        self.running_mean = (
            (self.counter["n"] * self.running_mean + other.counter["n"] * other.running_mean) / n if n > 0 else 0
        )

        delta = self.running_mean - other.running_mean
        self.running_variance = (
            self.running_variance + other.running_variance + self.counter["n"] * other.counter["n"] * delta**2 / n
            if n > 0
            else 0
        )

        self.max_value = max(self.max_value, other.max_value)
        self.min_value = min(self.min_value, other.min_value)
        self.counter["n"] = n
        return self

    def get_stats(self) -> OnlineStats:
        return OnlineStats(
            mean=self.running_mean,
            variance=self.running_variance,
            standard_deviation=math.sqrt(self.running_variance),
            total_time=self.counter["total"],
            count=self.counter["n"],
            max=self.max_value,
            min=self.min_value,
        )


class TimeStatsManager(ComputeOnlineStats):
    def __init__(self):
        super().__init__()

    def __enter__(self):
        self._entry_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._entry_time is not None
        self.update(time.perf_counter() - self._entry_time)
