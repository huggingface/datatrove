from dataclasses import dataclass, field
from typing import Generator, NewType


class MediaType:
    IMAGE = 0
    VIDEO = 1
    AUDIO = 2


@dataclass
class Media:
    """
    For possible future versions of datatrove
    """

    type: int
    url: str
    alt: str | None = None
    local_path: str | None = None


@dataclass
class Document:
    """
    Base datatrove data format.
    - `text` the actual text content for each sample
    - `id` a unique id (string) for this sample
    - `metadata` a dictionary where any additional info may be stored

    """

    text: str
    id: str
    media: list[Media] = field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


DocumentsPipeline = NewType("DocumentsPipeline", Generator[Document, None, None] | None)
