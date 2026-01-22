"""Data classes for the datatrove package."""

from dataclasses import dataclass, field
from typing import Any, Generator, NewType


class MediaType:
    """Media types

    For future uses, currently not used.
    """

    IMAGE = 0
    VIDEO = 1
    AUDIO = 2
    DOCUMENT = 3


@dataclass(slots=True)
class Media:
    """Media metadata

    For future uses, currently not used.
    """

    id: str
    type: int
    url: str
    alt: str | None = None
    path: str | None = None
    offset: int | None = None
    length: int | None = None
    media_bytes: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Document:
    """Main Document dataclass going through the processing pipeline

    Args:
        text: str
             the actual text content for each sample
        id: str
            a unique id (string) for this sample
        media: list[Media]
            The media associated with the document
        metadata: dict[str, Any]
            a dictionary where any additional info may be stored
    """

    text: str
    id: str
    media: list[Media] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


DocumentsPipeline = NewType("DocumentsPipeline", Generator[Document, None, None] | None)
