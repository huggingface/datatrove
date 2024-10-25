from typing import Literal

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import PRECALCULATED_STATS, BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.lid import FT176LID, GlotLID
from datatrove.utils.logging import logger


class LanguageFilter(BaseFilter):
    name = "🌍 Language ID"
    _requires_dependencies = [("fasttext", "fasttext-wheel"), "fasteners"]

    def __init__(
        self,
        precalculated_stats: PRECALCULATED_STATS = PRECALCULATED_STATS.re_calculate_if_missing,
        languages: list[str] | str | None = None,
        language_threshold: float = 0.65,
        exclusion_writer: DiskWriter = None,
        backend: Literal["ft176", "glotlid"] = "ft176",
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
        self.precalculated_stats = precalculated_stats
        self.language_threshold = language_threshold
        if isinstance(languages, str):
            languages = [languages]
        self.languages = languages
        self.backend = backend
        self.model = FT176LID(languages) if backend == "ft176" else GlotLID(languages)
        self.keep_top_pairs_threshold = keep_top_pairs_threshold

    def _filter_from_existing_stats(self, doc: Document) -> bool | tuple[bool, str]:
        if "language" not in doc.metadata or "language_score" not in doc.metadata:
            logger.warning(
                f"Missing 'language' in doc metadata for {doc.id}"
                "Ensure that the previous enricher war run with `language` enabled."
            )
            return False, "missing_language_field"

        lang_score = doc.metadata["language_score"]
        lang_pairs = {
            key.split("_")[-2]: value for key, value in doc.metadata.items() if key.startswith("top_language_")
        }
        return (self.languages and any(score > self.language_threshold for score in lang_pairs.values())) or (
            self.languages is None and lang_score > self.language_threshold
        )

    def _filter_maybe_from_existing_stats(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter
        """
        _force_recalc = False
        if self.precalculated_stats == PRECALCULATED_STATS.re_calculate:
            _force_recalc = True

        if "language" not in doc.metadata or "language_score" not in doc.metadata or _force_recalc:
            best_lang_pair, lang_pairs = self.model.predict(doc)
            lang, lang_score = best_lang_pair
            doc.metadata["language"] = lang
            doc.metadata["language_score"] = lang_score
        else:
            lang = doc.metadata["language"]
            lang_score = doc.metadata["language_score"]
            lang_pairs = {
                key.split("_")[-2]: value for key, value in doc.metadata.items() if key.startswith("top_language_")
            }

        if self.backend == "glotlid" and "language_script" not in doc.metadata:
            lang, script = lang.split("_")
            doc.metadata["language_script"] = script

        if self.keep_top_pairs_threshold != -1:
            for key, value in lang_pairs.items():
                if value > self.keep_top_pairs_threshold:
                    doc.metadata[f"top_language_{key}_score"] = value
        return (self.languages and any(score > self.language_threshold for score in lang_pairs.values())) or (
            self.languages is None and lang_score > self.language_threshold
        )

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        if (
            self.precalculated_stats == PRECALCULATED_STATS.re_calculate
            or self.precalculated_stats == PRECALCULATED_STATS.re_calculate_if_missing
        ):
            return self._filter_maybe_from_existing_stats(doc)
        elif self.precalculated_stats == PRECALCULATED_STATS.re_use:
            if "language" not in doc.metadata:
                logger.warning(
                    f"Missing 'language' in doc metadata for {doc.id}"
                    "Ensure that the previous enricher war run with `language` enabled."
                )
                return False, "missing_language_field"
            return self._filter_from_existing_stats(doc)
        else:
            return True
