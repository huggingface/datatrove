import random
import time

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class RandomTest(PipelineStep):
    def __init__(self, name):
        self.name = name

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        print(f"STARTING {self.name=}, {rank=}, {world_size=}")
        time.sleep(random.randint(2, 5))
        print(f"COMPLETED {self.name=}, {rank=}, {world_size=}")
