import os

from fasttext.FastText import _FastText

import urllib.request
from loguru import logger

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.utils.typeshelper import Languages, LocalPaths, NiceRepr

LANGUAGE_ID_MODEL_URL = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'


class LanguageFilter(BaseFilter):

    def __init__(
            self,
            language_threshold: float = 0.65,
            model_local_path: str = "".join([LocalPaths.download, "/language_id/lid.176.bin"]),
            languages: tuple = (
                    Languages.english,
                    Languages.italian,
                    Languages.spanish,
                    Languages.portuguese,
                    Languages.french,
                    Languages.german,
                    Languages.romanian,
            ),
            **kwargs
    ):
        """
          filters if the predicted language is not among given language or if the language score is below language
          language_threshold

          @param regex_exp: language_threshold minimum score to accept a document.
          @param languages: list of languages to not filter out.
          """
        super().__init__(**kwargs)
        self.language_threshold = language_threshold
        self.languages = languages
        self.model_local_path = model_local_path
        self._model = None

    def __repr__(self):
        return " ".join([super().__repr__(), NiceRepr("🌍", "Language ID").get_name()])

    @property
    def model(self):
        if not self._model:
            if not os.path.isfile(self.model_local_path):
                os.makedirs(os.path.dirname(self.model_local_path), exist_ok=True)
                logger.info("⬇️ Downloading fast-text langauge identifier model ...")
                urllib.request.urlretrieve(LANGUAGE_ID_MODEL_URL, self.model_local_path)
            self._model = _FastText(self.model_local_path)
        return self._model

    def filter(self, doc: Document) -> bool:
        """

        :param doc: document
        :return: is_filter
        """

        language, score = self.model.predict(doc.content.replace("\n", ""))
        # language label is given in the form __label__<language_id>
        language = language[0].split("__")[2]
        doc.metadata["language_id"] = language
        doc.metadata["score"] = score[0]
        if score > self.language_threshold and language in self.languages:
            return True
        return False
