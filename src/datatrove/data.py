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


@dataclass
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
    media_bytes: bytes | None = None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self):
        """Decode base64-encoded media_bytes if needed.

        When Media objects are deserialized from JSONL, media_bytes may be a
        base64-encoded string instead of bytes. This converts it back.
        """
        if self.media_bytes is not None and isinstance(self.media_bytes, str):
            import base64
            self.media_bytes = base64.b64decode(self.media_bytes)

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
        metadata: dict[str, Any]
            a dictionary where any additional info may be stored
    """

    text: str
    id: str
    media: list[Media] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


DocumentsPipeline = NewType("DocumentsPipeline", Generator[Document, None, None] | None)
