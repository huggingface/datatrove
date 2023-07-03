from dataclasses import dataclass, field
from typing import Generator, NewType


class MediaType:
    IMAGE = 0
    VIDEO = 1
    AUDIO = 2


@dataclass
class Media:
    type: int
    url: str
    alt: str = None
    local_path: str = None


@dataclass
class Document:
    content: str
    data_id: str
    media: list[Media] = field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


DocumentsPipeline = NewType("DocumentsPipeline", Generator[Document, None, None] | None)
