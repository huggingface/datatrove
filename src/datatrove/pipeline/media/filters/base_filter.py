from abc import ABC, abstractmethod
from typing import Tuple

from datatrove.data import DocumentsPipeline, Media
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints


class BaseMediaContentFilter(PipelineStep, ABC):
    """Base module for Media Filters"""

    type = "🔻️ - MEDIA FILTER"

    def __init__(self):
        super().__init__()

    @abstractmethod
    def filter(self, media: Media) -> bool | Tuple[bool, str]:
        """Filter modules main method, for a single media
        Returns true if bytes should be KEPT, false if they should be REMOVED.

        Args:
            media: sample to filter

        Returns:
            bool - whether the media should be kept
            or (False, str), to drop with a specific reason
        """
        raise NotImplementedError

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        def get_filter_result(res):
            if isinstance(res, bool):
                return res, None
            return res[0], res[1]

        with self.track_time():
            for doc in data:
                for media in doc.media:
                    self.stat_update(StatHints.total)
                    filter_result, reason = get_filter_result(self.filter(media))
                    if not filter_result:
                        self.stat_update(StatHints.dropped)
                        media.media_bytes = None
                        media.metadata["filter_reason"] = reason
                        if reason:
                            self.stat_update(f"dropped_{reason}")
                    else:
                        self.stat_update(StatHints.forwarded)
                yield doc
