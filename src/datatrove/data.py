"""Data classes for the datatrove package."""

from dataclasses import dataclass, field
from typing import Generator, NewType


class MediaType:
    """Media types

    For future uses, currently not used.
    """

    IMAGE = 0
    VIDEO = 1
    AUDIO = 2


@dataclass
class Media:
    """Media metadata

    For future uses, currently not used.
    """

    type: int
    url: str
    alt: str | None = None
    local_path: str | None = None


@dataclass
class Document:
    """Main Document dataclass going through the processing pipeline

    Args:
        text: str
             the actual text content for each sample
        id: str
            a unique id (string) for this sample
        media: list[Media]
            The media associated with the document
        metadata: dict[str, str | int | float | bool]
            a dictionary where any additional info may be stored
    """

    text: str
    id: str
    media: list[Media] = field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


DocumentsPipeline = NewType("DocumentsPipeline", Generator[Document, None, None] | None)
