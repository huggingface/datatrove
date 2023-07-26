from trafilatura import extract

from .base import BaseExtractor


class Trafilatura(BaseExtractor):
    """
    Trafilatura extractor, it uses https://trafilatura.readthedocs.io/en/latest/index.html
    """

    name = "⛏️ Trafilatura"

    def __init__(self, favour_precision: bool = False, include_images: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.favour_precision = favour_precision
        self.include_images = include_images
        if self.include_images:
            raise NotImplementedError

    def extract(self, content: str) -> str:
        return extract(content)
