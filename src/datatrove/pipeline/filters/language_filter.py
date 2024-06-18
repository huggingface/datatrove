from typing import Literal

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.lid import FT176LID, GlotLID


class LanguageFilter(BaseFilter):
    name = "🌍 Language ID"
    _requires_dependencies = [("fasttext", "fasttext-wheel"), "fasteners"]

    def __init__(
        self,
        languages: list[str] | str = None,
        language_threshold: float = 0.65,
        exclusion_writer: DiskWriter = None,
        backend: Literal["ft176", "glotlid"] = "ft176",
        label_only: bool = False,
    ):
        """
        filters if the predicted language is not among given language or if the language score is below language
        language_threshold

        Args:
            languages: list of languages to keep. None for all
            language_threshold: language_threshold minimum score to accept a document
            exclusion_writer:
            label_only: if True, only the language label is added to the metadata and no documents are removed
        """
        super().__init__(exclusion_writer)
        self.language_threshold = language_threshold
        if isinstance(languages, str):
            languages = list(languages)
        self.languages = languages
        self.backend = backend
        self.model = FT176LID(languages) if backend == "ft176" else GlotLID(languages)
        self.label_only = label_only

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter
        """
        best_lang_pair, lang_pairs = self.model.predict(doc)
        lang, lang_score = best_lang_pair
        if self.backend == "glotlid":
            lang, script = lang.lower().split("_")
            doc.metadata["language_script"] = script
        doc.metadata["language"] = lang
        doc.metadata["language_score"] = lang_score
        return (
            self.label_only
            or (self.languages and any(score > self.language_threshold for score in lang_pairs.values()))
            or (self.languages is None and lang_score > self.language_threshold)
        )
