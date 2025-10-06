from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PageResponse:
    natural_text: Optional[str]
    finish_reason: Optional[str]

@dataclass(frozen=True)
class PageResult:
    page_num: int
    response: PageResponse
    input_tokens: int
    output_tokens: int
    failed: bool
    image_base64: Optional[str] = None