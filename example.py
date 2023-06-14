from src.datatrove.executor.base import PipelineExecutor

pipeline = [
    Reader(),
    dfsdf(),
    Writer()
]


executor: PipelineExecutor = Executor(
    ...,
    pipeline=pipeline
)

executor.run()







