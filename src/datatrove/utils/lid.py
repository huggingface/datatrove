from abc import abstractmethod

from datatrove.data import Document
from datatrove.io import cached_asset_path_or_download


class LID:
    def __init__(self, languages: list[str]) -> None:
        self.languages = languages

    @abstractmethod
    def predict(self, doc: Document) -> tuple[tuple[str, int], dict[str, float]]:
        """
        Predicts the likelihood of the document being written in given languages, alongside with the most likely language
        Args:
            doc (Document): Document to predict languages for
        Returns:
            dict[str, float]: Languages and score
        """
        raise NotImplementedError


class FastTextModel(LID):
    LANGUAGE_ID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    def __init__(self, languages: list[str], k: int = 1) -> None:
        """
        Args:
            languages (list[str]): Languages to predict
            k (int, optional): Number of top-k languages to consider, all languages outside of k will be considered as being predicted with 0.0
        """
        super().__init__(languages)
        self._model = None
        self.k = k

    @property
    def model(self):
        if not self._model:
            from fasttext.FastText import _FastText

            model_file = cached_asset_path_or_download(
                self.LANGUAGE_ID_MODEL_URL,
                namespace="filters",
                subfolder="language_filter",
                desc="fast-text language identifier model",
            )
            self._model = _FastText(model_file)
        return self._model

    def predict(self, doc: Document) -> tuple[tuple[str, int], dict[str, float]]:
        langs, scores = self.model.predict(doc.text.replace("\n", " "), k=self.k)
        lang_pairs = {lang.split("__")[2]: score for lang, score in zip(langs, scores)}
        best_lang_pair = max(lang_pairs.items(), key=lambda x: x[1])
        return best_lang_pair, {lang: lang_pairs.get(lang, 0.0) for lang in self.languages}


# We don't support CLD3, not only it's worse than fasttext, but installation is really problematic, because of old version of protobuffers
