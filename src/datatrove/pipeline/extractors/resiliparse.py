from .base import BaseExtractor


class Resiliparse(BaseExtractor):
    """
    Resiliparse extractor, it uses https://resiliparse.chatnoir.eu/en/latest/index.html

    We're actually only using the main entry point of resiliparse's text extraction: the `extract_plain_text` function.
    No specific data structure is exchanged with Resiliparse, only the text is passed and the extracted text is returned.

    Args:
        timeout: the timeout for extraction, per document, in seconds
        preserve_formatting: whether to preserve the formatting of the text
        main_content: whether to extract the main content of the document
        list_bullets: whether to extract the bullets of the document
        alt_texts: whether to extract the alt texts of the document
        links: whether to extract the links of the document
        form_fields: whether to extract the form fields of the document
        noscript: whether to extract the noscript of the document
        comments: whether to extract the comments that are present in the document
        skip_elements: whether to skip the elements of the document
    """

    name = "⛏ Resiliparse"
    _requires_dependencies = ["resiliparse"]

    def __init__(
        self,
        preserve_formatting: bool = True,
        main_content: bool = True,
        list_bullets: bool = True,
        alt_texts: bool = False,
        links: bool = False,
        form_fields: bool = False,
        noscript: bool = False,
        comments: bool = True,
        skip_elements: list = None,
        timeout: float = 0.1,
        **kwargs,
    ):
        super().__init__(timeout)
        self.preserve_formatting = preserve_formatting
        self.main_content = main_content
        self.list_bullets = list_bullets
        self.alt_texts = alt_texts
        self.links = links
        self.form_fields = form_fields
        self.noscript = noscript
        self.comments = comments
        self.skip_elements = skip_elements

    def extract(self, text: str) -> str:
        """

        Args:
          text: str: html content

        Returns: plaintext extracted text

        """
        from resiliparse.extract.html2text import extract_plain_text

        return extract_plain_text(
            text,
            preserve_formatting=self.preserve_formatting,
            main_content=self.main_content,
            list_bullets=self.list_bullets,
            alt_texts=self.alt_texts,
            links=self.links,
            form_fields=self.form_fields,
            noscript=self.noscript,
            comments=self.comments,
            skip_elements=self.skip_elements,
        )
