import time
from tqdm import tqdm
import multiprocessing
from queue import Queue
from functools import partial
from threading import Thread
from multiprocessing import Pool, Process, Manager
from typing_extensions import TypeAlias


QueueType: TypeAlias = "Queue[Union[None, int]]"


class ThreadedProgressBar:


    def __init__(self, desc=None, total=None, disable=False, unit='it',
                 smoothing=0.3, bar_format=None, initial=0,
                 position=None, pbar_timeout=1e-3, worker_type=Thread, queue=None, **kwargs):
        self.desc = desc
        self.total = total
        self.disable = disable
        self.unit = unit
        self.smoothing = smoothing
        self.bar_format = bar_format
        self.initial = initial
        self.position = position
        self.kwargs = kwargs

        self.pbar = None
        self.pbar_timeout = pbar_timeout
        self._worker_type = worker_type
        self.manager = None
        self.pbar_queue = queue


    @classmethod
    def _run_threaded_progressbar(
        cls,
        queue: QueueType,
        timeout: float,
        pbar_args: dict,
    ):
        """Run a progress bar in a separate thread.

        Args:
            queue (QueueType): The queue to increment the progress bars.
            timeout (float): How often to update the progress bars in seconds.
            pbar_args (dict): 
        """

        with tqdm(**pbar_args) as pbar:
            while True:
                item = queue.get()
                if item is None:
                    break
                pbar.update(item)
                time.sleep(timeout)


    def __enter__(self):
        pbar_args = dict(
            desc=self.desc,
            total=self.total,
            disable=self.disable,
            unit=self.unit,
            smoothing=self.smoothing,
            bar_format=self.bar_format,
            initial=self.initial,
            position=self.position,
            **self.kwargs,
            )
        if self.pbar_queue is None:
            self.manager = multiprocessing.Manager()
            self.pbar_queue: QueueType = self.manager.Queue()
        self.pbar_update_worker = self._worker_type(
            target=self._run_threaded_progressbar, args=(self.pbar_queue, self.pbar_timeout, pbar_args), daemon=True
        )
        self.pbar_update_worker.start()
        return self


    def update(self, value: int):
        self.pbar_queue.put(value)


    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


    def close(self):
        self.pbar_queue.put(None)
        self.pbar_update_worker.join()
        if self.manager is not None:
            self.manager.shutdown()
