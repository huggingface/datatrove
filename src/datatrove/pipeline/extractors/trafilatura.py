from trafilatura import extract

from .base import BaseExtractor


class Trafilatura(BaseExtractor):
    """Trafilatura extractor, it uses https://trafilatura.readthedocs.io/en/latest/index.html"""

    name = "⛏️ Trafilatura"

    def __init__(self, favour_precision: bool = False, include_images: bool = False, timeout: float = 0.1, **kwargs):
        """

        :param favour_precision: prefer less text but correct extraction.
        :param include_images: not implemented currently
        :param timeout: the timeout for extraction, per document, in seconds
        :param kwargs: any other option will be passed to trafilatura
        """
        super().__init__(timeout)
        self.favour_precision = favour_precision
        self.include_images = include_images
        self.kwargs = kwargs
        if self.include_images:
            raise NotImplementedError

    def extract(self, content: str) -> str:
        return extract(content, favor_precision=self.favour_precision, **self.kwargs)
