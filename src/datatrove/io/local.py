import os
from dataclasses import dataclass, field

from datatrove.io.base import InputDataFile, InputDataFolder, OutputDataFile, OutputDataFolder


@dataclass
class LocalOutputDataFolder(OutputDataFolder):
    local_path: str = field(init=False)

    def __post_init__(self):
        self.local_path = self.path

    def create_new_file(self, relative_path: str):
        return OutputDataFile(
            local_path=os.path.join(self.local_path, relative_path),
            path=os.path.join(self.path, relative_path),
            relative_path=relative_path,
        )


@dataclass
class LocalInputDataFolder(InputDataFolder):
    def list_files(self, extension: str | list[str] = None) -> list[InputDataFile]:
        return [
            InputDataFile(path, path)
            for path in get_local_file_list(self.path, self.recursive)
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
