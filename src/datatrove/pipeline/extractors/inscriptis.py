import re

from .base import BaseExtractor


class Inscriptis(BaseExtractor):
    """Inscriptis extractor, it uses https://github.com/weblyzard/inscriptis

    We're using the main entry point of inscriptis: the `get_text` function.
    No specific data structure is exchanged with Inscriptis, only the HTML is passed and the extracted text is returned.

    Args:
        timeout: the timeout for extraction, per document, in seconds
        deduplicate_captions: whether to remove duplicate captions
        display_links: whether to display link targets
        display_anchors: whether to display anchor texts
        **kwargs: any other option will be passed to inscriptis
    """

    name = "⛏ Inscriptis"
    _requires_dependencies = ["inscriptis"]

    def __init__(
        self,
        preprocessor: BaseExtractor,
        timeout: float = 0.1,
        max_new_lines: int = 2,
        deduplicate_captions: bool = True,
        display_links: bool = False,
        display_anchors: bool = True,
        **kwargs,
    ):
        super().__init__(timeout)
        self.preprocessor = preprocessor
        self.new_line_chars = "\n" * max_new_lines
        self.deduplicate_captions = deduplicate_captions
        self.display_links = display_links
        self.display_anchors = display_anchors
        self.kwargs = kwargs
        self.regex_excessive_lines = re.compile(r"(" + self.new_line_chars + "\n+)")

    def clean_html(self, html: str) -> str:
        return self.preprocessor.clean_html(html)

    def extract(self, text: str) -> str:
        """
        Args:
          text: str: html content

        Returns: plaintext extracted text
        """
        from inscriptis import get_text
        from inscriptis.css_profiles import CSS_PROFILES
        from inscriptis.model.config import ParserConfig

        cleaned_html = self.clean_html(text)

        text = get_text(
            html_content=cleaned_html,
            config=ParserConfig(
                css=CSS_PROFILES["strict"],
                deduplicate_captions=self.deduplicate_captions,
                display_links=self.display_links,
                display_anchors=self.display_anchors,
                **self.kwargs,
            ),
        )

        # remove excessive empty lines
        return self.regex_excessive_lines.sub(self.new_line_chars, text).strip()
