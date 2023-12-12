from dataclasses import dataclass, field
from typing import Dict, Generator, List, NewType, Optional, Union


class MediaType:
    IMAGE = 0
    VIDEO = 1
    AUDIO = 2


@dataclass
class Media:
    type: int
    url: str
    alt: Optional[str] = None
    local_path: Optional[str] = None


@dataclass
class Document:
    content: str
    data_id: str
    media: List[Media] = field(default_factory=list)
    metadata: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)


DocumentsPipeline = NewType("DocumentsPipeline", Optional[Generator[Document, None, None]])
