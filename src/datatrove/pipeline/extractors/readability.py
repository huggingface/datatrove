from .base import BaseExtractor


class Readability(BaseExtractor):
    """Readability extractor, it uses https://github.com/buriy/python-readability

    We're using the main entry point of readability-lxml: the `Document` class.
    No specific data structure is exchanged with Readability, only the HTML is passed and the extracted text is returned.

    Args:
        timeout: the timeout for extraction, per document, in seconds
        min_text_length: the minimum length of text to consider
        retry_length: number of chars to use when searching for body
        url: the URL of the page (optional, used for better parsing)
        keep_classes: list of classes to keep in the extracted content
        **kwargs: any other option will be passed to readability
    """

    name = "⛏ Readability"
    _requires_dependencies = ["readability"]

    def __init__(
        self,
        timeout: float = 0.1,
        min_text_length: int = 25,
        retry_length: int = 250,
        url: str = None,
        **kwargs,
    ):
        super().__init__(timeout)
        self.min_text_length = min_text_length
        self.retry_length = retry_length
        self.url = url
        self.kwargs = kwargs

    def extract(self, text: str, postprocessor: BaseExtractor) -> str:
        """
        Args:
          text: str: html content

        Returns: plaintext extracted text
        """
        from readability import Document

        if not postprocessor:
            raise ValueError("A postprocessor (extractor) must be provided")

        doc = Document(
            text,
            min_text_length=self.min_text_length,
            retry_length=self.retry_length,
            url=self.url,
            **self.kwargs,
        )

        cleaned_html = doc.summary()

        return postprocessor.extract(cleaned_html)
