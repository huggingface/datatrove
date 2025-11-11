import magic

from datatrove.data import Media
from datatrove.pipeline.media.filters.base_filter import BaseMediaContentFilter


class MimeTypeFilter(BaseMediaContentFilter):
    name = "🔍 Mime Type Filter"

    def __init__(self, mime_types: list[str]):
        super().__init__()
        self.mime_types = mime_types

    def filter(self, media: Media):
        if media.media_bytes is None:
            return False, "media_bytes_is_none"

        content_mime_type = magic.from_buffer(media.media_bytes, mime=True)
        if content_mime_type not in self.mime_types:
            return False, f"content_mime_type_{content_mime_type.lower()}"
        return True
