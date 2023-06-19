import os

from fasttext.FastText import _FastText

import urllib.request

from datatrove.data import Document
from datatrove.pipeline.filters.base import BaseFilter
from datatrove.utils.typeshelper import Languages

LANGUAGE_ID_MODEL_URL = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'


class LanguageID(BaseFilter):

    def __init__(
            self,
            language_threshold: float = 0.65,
            model_local_path: str = "/somewhere/language_id",
            languages: tuple = (
                    Languages.english,
                    Languages.italian,
                    Languages.spanish,
                    Languages.portuguese,
                    Languages.french,
                    Languages.german,
                    Languages.romanian
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
        if not os.path.isfile(model_local_path):
            urllib.request.urlretrieve(LANGUAGE_ID_MODEL_URL, model_local_path)
        self.model = _FastText(model_local_path)

    def __repr__(self):
        return " ".join([super().__repr__(), "🌍 langauge id"])

    def filter(self, doc: Document) -> bool:
        """

        :param doc: document
        :return: is_filter
        """

        language, score = self.model.predict(doc.content)
        doc.metadata["language_id"] = language
        doc.metadata["score"] = language
        if score > self.language_threshold and language in self.languages:
            return True
        return False
