from .base import BaseExtractor


class Justext(BaseExtractor):
    """Justext extractor, it uses https://github.com/miso-belica/jusText

    We're actually only using the main entry point of justext: the `justext` function.
    No specific data structure is exchanged with Justext, only the text is passed and the extracted text is returned.

    Args:
        length_low: the minimum length of a paragraph
        length_high: the maximum length of a paragraph
        stopwords_low: the minimum stopwords ratio of a paragraph
        stopwords_high: the maximum stopwords ratio of a paragraph
        max_link_density: the maximum link density of a paragraph
        max_heading_distance: the maximum distance between headings of a paragraph
        no_headings: whether to remove headings from the extracted text
        remove_boilerplate: whether to remove boilerplate from the extracted text
        kwargs: any other option will be passed to justext
        timeout: the timeout for extraction, per document, in seconds
    """

    name = "⛏ Justext"
    _requires_dependencies = ["justext"]

    def __init__(
        self,
        stoplist: list[str] = None,
        length_low: int = 70,
        length_high: int = 200,
        stopwords_low: float = 0.3,
        stopwords_high: float = 0.32,
        max_link_density: float = 0.2,
        max_heading_distance: int = 200,
        no_headings: bool = False,
        remove_boilerplate: bool = True,
        timeout: float = 0.1,
        **kwargs,
    ):
        super().__init__(timeout)
        if stoplist is None:
            stoplist = self.get_stoplist()
        self.stoplist = frozenset(stoplist)
        self.length_low = length_low
        self.length_high = length_high
        self.stopwords_low = stopwords_low
        self.stopwords_high = stopwords_high
        self.max_link_density = max_link_density
        self.max_heading_distance = max_heading_distance
        self.no_headings = no_headings
        self.remove_boilerplate = remove_boilerplate
        self.kwargs = kwargs

    @staticmethod
    def get_stoplist(lang: str = "English") -> list[str]:
        from justext import get_stoplist

        return get_stoplist(lang)

    def clean_html(self, html: str) -> str:
        """

        Args:
          html: str: html content

        Returns: cleaned HTML
        """
        from justext.core import html_to_dom, preprocessor
        from lxml.html import tostring

        dom = html_to_dom(html)
        dom = preprocessor(dom)
        cleaned_html = tostring(dom).decode()
        return cleaned_html

    def extract(self, text: str) -> str:
        """

        Args:
          text: str: html content

        Returns: plaintext extracted text
        """
        from justext import justext

        paragraphs = justext(
            text,
            stoplist=self.stoplist,
            length_low=self.length_low,
            length_high=self.length_high,
            stopwords_low=self.stopwords_low,
            stopwords_high=self.stopwords_high,
            max_link_density=self.max_link_density,
            max_heading_distance=self.max_heading_distance,
            no_headings=self.no_headings,
            **self.kwargs,
        )

        # Join text blocks with double newlines to separate them
        if self.remove_boilerplate:
            return "\n\n".join([p.text for p in paragraphs if not p.is_boilerplate])
        else:
            return "\n\n".join([p.text for p in paragraphs])
