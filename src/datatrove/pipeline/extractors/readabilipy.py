from .base import BaseExtractor


class ReadabiliPy(BaseExtractor):
    """ReadabiliPy extractor, it uses https://github.com/alan-turing-institute/ReadabiliPy

    We're using the main entry point of ReadabiliPy: the `simple_json_from_html_string` function.
    The extracted content is returned as plain text.

    Args:
        timeout: the timeout for extraction, per document, in seconds
        use_readability: whether to use Mozilla's Readability.js (requires Node.js)
        content_digests: whether to include content digests in the output
        node_indexes: whether to include node indexes in the output
        **kwargs: any other option will be passed to ReadabiliPy
    """

    name = "⛏ ReadabiliPy"
    _requires_dependencies = ["readabilipy"]

    def __init__(
        self,
        timeout: float = 0.1,
        use_readability: bool = False,
        content_digests: bool = False,
        node_indexes: bool = False,
        **kwargs,
    ):
        super().__init__(timeout)
        self.use_readability = use_readability
        self.content_digests = content_digests
        self.node_indexes = node_indexes
        self.kwargs = kwargs

    def extract(self, text: str) -> str:
        """
        Args:
          text: str: html content

        Returns: plaintext extracted text
        """
        from readabilipy import simple_json_from_html_string

        result = simple_json_from_html_string(
            text,
            use_readability=self.use_readability,
            content_digests=self.content_digests,
            node_indexes=self.node_indexes,
            **self.kwargs,
        )

        content = result.get("plain_text", "")

        if isinstance(content, list):
            content = "\n\n".join(block["text"] for block in content)

        return content
