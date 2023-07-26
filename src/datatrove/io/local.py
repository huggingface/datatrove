import os
from dataclasses import dataclass, field

from datatrove.io.base import BaseInputDataFolder, BaseOutputDataFolder, InputDataFile, OutputDataFile


@dataclass
class LocalOutputDataFolder(BaseOutputDataFolder):
    local_path: str = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.local_path = self.path

    def create_new_file(self, relative_path: str):
        return OutputDataFile(
            local_path=os.path.join(self.local_path, relative_path),
            path=os.path.join(self.path, relative_path),
            relative_path=relative_path,
        )


@dataclass
class LocalInputDataFolder(BaseInputDataFolder):
    def list_files(self, extension: str | list[str] = None, suffix: str = "") -> list[InputDataFile]:
        return [
            InputDataFile(path)
            for path in get_local_file_list(os.path.join(self.path, suffix), self.recursive)
            if self._match_file(path, extension)
        ]


def get_local_file_list(path: str, recursive: bool = True) -> list[str]:
    filelist = []
    with os.scandir(path) as files:
        for f in files:
            if f.is_file():
                filelist.append(os.path.abspath(f.path))
            elif recursive:
                filelist.extend(get_local_file_list(f.path, recursive=recursive))
    return sorted(filelist)
