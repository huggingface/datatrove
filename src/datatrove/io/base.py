import contextlib
import os
from abc import abstractmethod, ABC
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from fnmatch import fnmatch


@dataclass
class InputDataFile:
    local_path: str
    path: str

    @contextmanager
    def open(self, open_fn: Callable = None):
        with open(self.local_path) if not open_fn else open_fn(self.local_path) as f:
            yield f


@dataclass
class InputDataFolder(ABC):
    path: str
    extension: str | list[str] = None
    recursive: bool = True
    match_pattern: str = None

    @abstractmethod
    def list_files(self) -> list[InputDataFile]:
        raise NotImplementedError

    def __post_init__(self):
        self._lock = contextlib.nullcontext()

    def set_lock(self, lock):
        self._lock = lock

    def get_files_shard(self, rank: int, world_size: int) -> list[InputDataFile]:
        return self.list_files()[rank::world_size]

    def _match_file(self, file_path):
        extensions = [self.extension] if type(self.extension) == str else self.extension
        return (  # check extension
                not extensions or os.path.splitext(file_path)[1] in extensions
        ) and (  # check pattern
                not self.match_pattern or fnmatch(os.path.relpath(file_path, self.path), self.match_pattern)
        )


@dataclass
class OutputDataFile(ABC):
    local_path: str
    path: str
    relative_path: str
    file_handler = None
    nr_documents: int = 0

    def close(self, close_fn: Callable = None):
        if self.file_handler:
            (close_fn or (lambda x: x.close()))(self.file_handler)

    def open(self, open_fn: Callable = None):
        os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
        self.file_handler = open(self.local_path, "w") if not open_fn else open_fn(self.local_path)
        self.file_handler = open_fn(self.local_path)
        return self.file_handler


@dataclass
class OutputDataFolder(ABC):
    path: str
    local_path: str
    _output_files: dict[str, OutputDataFile] = field(default_factory=dict)

    def close(self, close_fn: Callable = None):
        for file in self._output_files.values():
            file.close(close_fn=close_fn)

    @abstractmethod
    def create_new_file(self, relative_path: str):
        raise NotImplementedError

    def __post_init__(self):
        self._lock = contextlib.nullcontext()

    def set_lock(self, lock):
        self._lock = lock

    def get_file(self, relative_path: str, open_fn: Callable = None):
        if relative_path not in self._output_files:
            new_output_file = self.create_new_file(relative_path)
            new_output_file.open(open_fn)
            self._output_files[relative_path] = new_output_file
        return self._output_files[relative_path]
