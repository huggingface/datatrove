import os
import urllib.request

from huggingface_hub import cached_assets_path
from loguru import logger

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.assets import DOWNLOAD_PATH
from datatrove.utils.typeshelper import Languages


LANGUAGE_ID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

FASTTEXT_INSTALLED = True
try:
    from fasttext.FastText import _FastText
except ImportError:
    FASTTEXT_INSTALLED = False


class LanguageFilter(BaseFilter):
    name = "🌍 Language ID"

    def __init__(
        self,
        languages: tuple = (Languages.english,),
        language_threshold: float = 0.65,
        model_local_path: str = os.path.join(DOWNLOAD_PATH, "language_id/lid.176.bin"),
        exclusion_writer: DiskWriter = None,
    ):
        """
        filters if the predicted language is not among given language or if the language score is below language
        language_threshold

        @param regex_exp: language_threshold minimum score to accept a document.
        @param languages: list of languages to not filter out.
        """
        super().__init__(exclusion_writer)
        self.language_threshold = language_threshold
        self.languages = languages
        self._model = None

    @property
    def model(self):
        if not self._model:
            if not FASTTEXT_INSTALLED:
                logger.error("FastText is required to run LanguageFilter")
                raise ImportError
            download_dir = cached_assets_path("datatrove", "filters", "language_filter")
            model_file = os.path.join(download_dir, "lid.176.bin")
            if not os.listdir(download_dir):
                logger.info("⬇️ Downloading fast-text language identifier model ...")
                urllib.request.urlretrieve(LANGUAGE_ID_MODEL_URL, model_file)
            self._model = _FastText(model_file)
        return self._model

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter
        """

        language, score = self.model.predict(doc.content.replace("\n", ""))
        # language label is given in the form __label__<language_id>
        language = language[0].split("__")[2]
        doc.metadata["language"] = language
        doc.metadata["language_score"] = score[0]
        return score > self.language_threshold and language in self.languages
