import os

from huggingface_hub import cached_assets_path
from loguru import logger

from datatrove.io import download_file
from datatrove.pipeline.base import DocumentsPipeline, PipelineStep


LANGUAGE_ID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


class LanguageIdStats(PipelineStep):
    """
    Pipeline step to identify the language of each document in a pipeline.
    Utilizes the fasttext supervised model for language identification and adds the identified language
    and its confidence score as metadata to each document.

    The identified language is stored under the 'language' key, and the confidence score under 'language_score'.
    This step requires the fasttext library and downloads the necessary model if not present.

    Requires:
        fasttext-wheel: A dependency required for the fasttext model to function properly.
    """

    type = "📊 - STATS"
    name = "🌍 Language ID"
    _requires_dependencies = [("fasttext", "fasttext-wheel")]

    def __init__(
        self,
    ):
        super().__init__()
        self._model = None

    @property
    def model(self):
        if not self._model:
            from fasttext.FastText import _FastText

            download_dir = cached_assets_path(
                # We need to discuss where to put the language model.
                library_name="datatrove",
                namespace="filters",
                subfolder="language_filter",
            )
            model_file = os.path.join(download_dir, "lid.176.bin")
            if not os.path.isfile(model_file):
                logger.info("⬇️ Downloading fast-text language identifier model...")
                download_file(LANGUAGE_ID_MODEL_URL, model_file)
                logger.info("⬇️ Downloaded fast-text language identifier model.")
            self._model = _FastText(model_file)
        return self._model

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            language, score = self.model.predict(doc.text.replace("\n", ""))
            # language label is given in the form __label__<language_id>
            language = language[0].split("__")[2]

            doc.metadata["language"] = language
            doc.metadata["language_score"] = score[0]

            self.stat_update(f"language_{language}", value=1, unit="doc")

            yield doc
