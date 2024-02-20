import os

from huggingface_hub import cached_assets_path
from loguru import logger

from datatrove.data import Document
from datatrove.io import download_file
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.typeshelper import Languages


LANGUAGE_ID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


class LanguageFilter(BaseFilter):
    name = "🌍 Language ID"
    _requires_dependencies = [("fasttext", "fasttext-wheel")]

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
        self._model = None

    @property
    def model(self):
        if not self._model:
            from fasttext.FastText import _FastText

            download_dir = cached_assets_path(
                library_name="datatrove", namespace="filters", subfolder="language_filter"
            )
            model_file = os.path.join(download_dir, "lid.176.bin")
            if not os.path.isfile(model_file):
                logger.info("⬇️ Downloading fast-text language identifier model...")
                download_file(LANGUAGE_ID_MODEL_URL, model_file)
                logger.info("⬇️ Downloaded fast-text language identifier model.")
            self._model = _FastText(model_file)
        return self._model

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter
        """

        language, score = self.model.predict(doc.text.replace("\n", ""))
        # language label is given in the form __label__<language_id>
        language = language[0].split("__")[2]
        doc.metadata["language"] = language
        doc.metadata["language_score"] = score[0]
        return score > self.language_threshold and language in self.languages
