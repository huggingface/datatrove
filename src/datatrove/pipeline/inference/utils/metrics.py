import asyncio
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Set


class MetricsKeeper:
    def __init__(self, window=60 * 5):
        """
        Initializes the MetricsKeeper.

        Args:
            window (int): Time window in seconds for recent metrics. Defaults to 5 minutes.
        """
        self.window = window  # Time window in seconds
        self.start_time = time.time()  # Timestamp when MetricsKeeper was created
        self.total_metrics = defaultdict(int)  # Cumulative metrics since start
        self.window_metrics: Deque[Any] = deque()  # Deque to store (timestamp, metrics_dict)
        self.window_sum = defaultdict(int)  # Sum of metrics within the window

    def add_metrics(self, **kwargs):
        """
        Adds metrics to the keeper.

        Args:
            **kwargs: Arbitrary keyword arguments representing metric names and their values.
        """
        current_time = time.time()
        # Update cumulative metrics
        for key, value in kwargs.items():
            self.total_metrics[key] += value

        # Append current metrics with timestamp to the deque
        self.window_metrics.append((current_time, kwargs))

        # Update window sums
        for key, value in kwargs.items():
            self.window_sum[key] += value

        # Remove metrics that are outside the time window
        while self.window_metrics and self.window_metrics[0][0] < current_time - self.window:
            old_time, old_metrics = self.window_metrics.popleft()
            for key, value in old_metrics.items():
                self.window_sum[key] -= value
                if self.window_sum[key] <= 0:
                    del self.window_sum[key]  # Clean up to prevent negative counts

    def __str__(self):
        """
        Returns a formatted string of metrics showing rate per second since start and within the window.

        Returns:
            str: Formatted metrics string as a table.
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        window_time = min(self.window, elapsed_time) if elapsed_time > 0 else 1  # Prevent division by zero

        # Header
        header = f"{'Metric Name':<30} {'Lifetime (/sec)':>20} {'Recently (/sec)':>20}"
        separator = "-" * len(header)
        lines = [header, separator]

        # Sort metrics alphabetically for consistency
        for key in sorted(self.total_metrics.keys()):
            total = self.total_metrics[key]
            window = self.window_sum.get(key, 0)
            total_rate = total / elapsed_time if elapsed_time > 0 else 0
            window_rate = window / window_time if window_time > 0 else 0
            line = f"{key:<30} {total_rate:>20.2f} {window_rate:>20.2f}"
            lines.append(line)

        return "\n".join(lines)

