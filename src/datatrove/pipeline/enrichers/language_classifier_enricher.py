from typing import Literal

from datatrove.data import Document
from datatrove.pipeline.enrichers.base_enricher import BaseEnricher
from datatrove.utils.lid import FT176LID, GlotLID


class LanguageEnricher(BaseEnricher):
    name = "🌍 Language ID Enricher"
    _requires_dependencies = [("fasttext", "fasttext-wheel"), "fasteners"]

    def __init__(
        self,
        languages: list[str] | str | None = None,
        backend: Literal["ft176", "glotlid"] = "ft176",
        keep_top_pairs_threshold: float = -1,
    ):
        """
        Enricher to predict the language of a document
        Args:
            languages (list[str] | str | None): List of languages to predict
            backend (str): Backend to use for language prediction
            keep_top_pairs_threshold (float): Threshold to keep top pairs
        """
        super().__init__()
        if isinstance(languages, str):
            languages = list(languages)
        self.languages = languages
        self.backend = backend
        self.model = FT176LID(languages) if backend == "ft176" else GlotLID(languages)
        self.keep_top_pairs_threshold = keep_top_pairs_threshold

    def enrich(self, doc: Document) -> Document:
        """Args:
            doc: document

        Returns:
            doc: document with language metadata
        """
        best_lang_pair, lang_pairs = self.model.predict(doc)
        lang, lang_score = best_lang_pair
        if self.backend == "glotlid":
            lang, script = lang.split("_")
            doc.metadata["language_script"] = script
        doc.metadata["language"] = lang
        doc.metadata["language_score"] = lang_score
        if self.keep_top_pairs_threshold != -1:
            for key, value in lang_pairs.items():
                if value > self.keep_top_pairs_threshold:
                    doc.metadata[f"top_language_{key}_score"] = value
        return doc
