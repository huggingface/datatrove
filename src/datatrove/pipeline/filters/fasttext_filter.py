import os

from fsspec.core import strip_protocol
from huggingface_hub import cached_assets_path
from loguru import logger

from datatrove.data import Document
from datatrove.io import download_file
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


class FastTextFilter(BaseFilter):
    name = "🤖 fastText"
    _requires_dependencies = [("fasttext", "fasttext-wheel")]

    def __init__(
        self,
        model_url: str,
        exclusion_writer: DiskWriter = None,
    ):
        """
        filters if the predicted language is not among given language or if the language score is below language
        language_threshold

        Args:
            model_url: url to download the model
            exclusion_writer:
        """
        super().__init__(exclusion_writer)
        self.model_url = model_url
        self._model = None

    @property
    def model(self):
        if not self._model:
            from fasttext.FastText import _FastText

            download_dir = cached_assets_path(library_name="datatrove", namespace="filters", subfolder="fasttext")

            model_file = os.path.join(download_dir, strip_protocol(self.model_url).replace("/", "_"))
            if not os.path.isfile(model_file):
                logger.info(f'⬇️ Downloading fast-text model from "{self.model_url}"...')
                download_file(self.model_url, model_file)
                logger.info(f'⬇️ Downloaded fast-text model to "{model_file}".')
            self._model = _FastText(model_file)
        return self._model

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter
        """
        return True
        # language, score = self.model.predict(doc.text.replace("\n", ""))
        # # language label is given in the form __label__<language_id>
        # language = language[0].split("__")[2]
        # doc.metadata["language"] = language
        # doc.metadata["language_score"] = score[0]
        # return score > self.language_threshold and language in self.languages
