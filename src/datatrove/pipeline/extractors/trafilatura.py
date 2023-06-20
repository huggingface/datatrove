from datatrove.data import Document
from .base import BaseExtractor

from trafilatura import extract


class Trafilatura(BaseExtractor):
    """
    Trafilatura extractor, it uses https://trafilatura.readthedocs.io/en/latest/index.html
    """

    def __init__(
            self,
            favour_precision: bool = False,
            include_images: bool = False,
            timeout: float = 0.1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.timeout = timeout
        self.favour_precision = favour_precision
        self.include_images = include_images
        if self.include_images:
            raise NotImplemented

    def __repr__(self):
        " ".join([super().__repr__(), "trafilatura"])

    def extract(self, doc: Document) -> bool:
        content = extract(doc.content)
        if content:
            doc.content = content
            return True
        return False
