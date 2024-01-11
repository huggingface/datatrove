from abc import ABC, abstractmethod
from typing import NoReturn, Type

from datatrove.data import Document, DocumentsPipeline
from datatrove.utils._import_utils import _is_package_available
from datatrove.utils.stats import Stats


class PipelineStep(ABC):
    name: str = None
    type: str = None

    def __new__(cls, *args, **kwargs):
        requires_dependencies = getattr(cls, "requires_dependencies", None)
        if requires_dependencies:
            missing_dependencies = [
                dependency
                for dependency in requires_dependencies
                if not _is_package_available(dependency if not isinstance(dependency, tuple) else dependency[0])
            ]
            _raise_error_for_missing_dependencies(cls, missing_dependencies)
        return super().__new__(cls, *args, **kwargs)

    def __init__(self):
        self.stats = Stats(str(self))

    def stat_update(self, *labels, value: int = 1, unit: str = None):
        for label in labels:
            self.stats[label].update(value, unit)

    def update_doc_stats(self, document: Document):
        self.stats["doc_len"] += len(document.content)
        if token_count := document.metadata.get("token_count", None):
            self.stats["doc_len_tokens"] += token_count

    def track_time(self, unit: str = None):
        if unit:
            self.stats.time_stats.unit = unit
        return self.stats.time_stats

    def set_up_dl_locks(self, dl_lock, up_lock):
        pass

    def __repr__(self):
        return f"{self.type}: {self.name}"

    @abstractmethod
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if data:
            yield from data

    def __call__(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        return self.run(data, rank, world_size)


def _raise_error_for_missing_dependencies(
    pipeline_step: Type[PipelineStep], dependencies: list[tuple[str, ...]]
) -> NoReturn:
    """Raise an ImportError for the pipeline step's missing dependencies."""
    packages = [dependency if not isinstance(dependency, tuple) else dependency[0] for dependency in dependencies]
    pip_packages = [dependency if not isinstance(dependency, tuple) else dependency[1] for dependency in dependencies]
    if len(packages) > 1:
        packages = f"{','.join('`' + package_name + '`' for package_name in packages[:-1])} and `{packages[-1]}`"
    else:
        packages = f"`{packages[0]}`"
    raise ImportError(
        f"Please install {packages} to use pipeline step {type(pipeline_step.__name__)} (`pip install {' '.join(pip_packages)}`)."
    )
