from typing import Literal

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.lid import FT176LID, GlotLID


class LanguageFilter(BaseFilter):
    name = "🌍 Language ID"
    _requires_dependencies = [("fasttext", "fasttext-numpy2-wheel"), "fasteners"]

    def __init__(
        self,
        languages: list[str] | str | None = None,
        language_threshold: float = 0.65,
        exclusion_writer: DiskWriter = None,
        backend: Literal["ft176", "glotlid"] = "ft176",
        label_only: bool = False,
        keep_top_pairs_threshold: float = -1,
    ):
        """
        filters if the predicted language is not among given language or if the language score is below language
        language_threshold

        Args:
            languages: list of languages to keep. None for all
            language_threshold: language_threshold minimum score to accept a document
            exclusion_writer:
            label_only: if True, only the language label is added to the metadata and no documents are removed
            keep_top_pairs_threshold: keep a list of all language pairs with at least this score. -1 to disable
        """
        super().__init__(exclusion_writer)
        self.language_threshold = language_threshold
        if isinstance(languages, str):
            languages = [languages]
        self.languages = languages
        self.backend = backend
        self.model = FT176LID(languages) if backend == "ft176" else GlotLID(languages)
        self.label_only = label_only
        self.keep_top_pairs_threshold = keep_top_pairs_threshold

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter
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
        return (
            self.label_only
            or (self.languages and any(score > self.language_threshold for score in lang_pairs.values()))
            or (self.languages is None and lang_score > self.language_threshold)
        )
