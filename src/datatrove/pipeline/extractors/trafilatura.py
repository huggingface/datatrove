from .base import BaseExtractor


class Trafilatura(BaseExtractor):
    """Trafilatura extractor, it uses https://trafilatura.readthedocs.io/en/latest/index.html"""

    name = "⛏ Trafilatura"
    _requires_dependencies = ["trafilatura"]

    def __init__(
        self,
        favour_precision: bool = True,
        include_images: bool = False,
        timeout: float = 0.1,
        deduplicate: bool = True,
        **kwargs,
    ):
        """

        :param favour_precision: prefer less text but correct extraction.
        :param include_images: not implemented currently
        :param timeout: the timeout for extraction, per document, in seconds
        :param kwargs: any other option will be passed to trafilatura
        """
        super().__init__(timeout)
        self.favour_precision = favour_precision
        self.include_images = include_images
        self.deduplicate = deduplicate
        self.kwargs = kwargs
        if self.include_images:
            raise NotImplementedError

    def extract(self, text: str) -> str:
        """

        Args:
          text: str: html content

        Returns: plain text extracted text

        """
        from trafilatura import extract

        return extract(
            text,
            favor_precision=self.favour_precision,
            include_comments=False,
            deduplicate=self.deduplicate,
            **self.kwargs,
        )
