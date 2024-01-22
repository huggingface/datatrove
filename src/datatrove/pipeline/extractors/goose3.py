# Goose3 extractor for DataTrove - extract text with goose3
# mrfakename (www.mrfake.name)
# Apache License 2.0
from .base import BaseExtractor


class Goose3(BaseExtractor):
    """Goose3 extractor, it uses https://github.com/goose3/goose3"""

    name = "⛏ Goose3"
    _requires_dependencies = ["goose3"]

    def __init__(
        self,
		timeout: float = 0.1,
        include_images: bool = False,
		keep_footnotes: bool = True,
        **kwargs,
    ):
        """
		
        :param timeout: the timeout for extraction, per document, in seconds
        :param include_images: not implemented currently
        :param keep_footnotes: whether to keep footnotes
        :param kwargs: any other option will be passed to goose3
        """
        super().__init__(timeout)
        self.keep_footnotes = keep_footnotes
        self.include_images = include_images
        self.kwargs = kwargs
        self.g = None
        if self.include_images:
            raise NotImplementedError

    def extract(self, text: str) -> str:
        from goose3 import Goose
        if not self.g:
        	self.g = Goose()
        article = self.g.extract(raw_html=text, keep_footnotes=self.keep_footnotes **self.kwargs)
        return article.cleaned_text(
            text,
            favor_precision=self.favour_precision,
            include_comments=False,
            deduplicate=self.deduplicate,
        )
