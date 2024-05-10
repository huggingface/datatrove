from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.lid import FastTextModel
from datatrove.utils.typeshelper import Languages


LANGUAGE_ID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


class LanguageFilter(BaseFilter):
    name = "🌍 Language ID"
    _requires_dependencies = [("fasttext", "fasttext-wheel"), "fasteners"]

    def __init__(
        self,
        languages: tuple = (Languages.english,),
        language_threshold: float = 0.65,
        exclusion_writer: DiskWriter = None,
    ):
        """
        filters if the predicted language is not among given language or if the language score is below language
        language_threshold

        Args:
            languages: list of languages to keep
            language_threshold: language_threshold minimum score to accept a document
            exclusion_writer:
        """
        super().__init__(exclusion_writer)
        self.language_threshold = language_threshold
        self.languages = languages
        self.model = FastTextModel(list(languages))

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter
        """

        best_lang_pair, lang_pairs = self.model.predict(doc)
        doc.metadata["language"] = best_lang_pair[0]
        doc.metadata["language_score"] = best_lang_pair[1]
        return any(score > self.language_threshold for score in lang_pairs.values())
