import random
import time

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class RandomTest(PipelineStep):
    def __init__(self, name):
        self.name = name

    def __call__(self, rank: int, world_size: int, data: DocumentsPipeline) -> DocumentsPipeline:
        print(f"STARTING {self.name=}, {rank=}, {world_size=}")
        time.sleep(random.randint(2, 5))
        print(f"COMPLETED {self.name=}, {rank=}, {world_size=}")
