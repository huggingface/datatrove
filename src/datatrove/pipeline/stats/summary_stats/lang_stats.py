from abc import abstractmethod
from typing import get_args

from datatrove.data import Document
from datatrove.io import DataFolderLike, cached_asset_path_or_download
from datatrove.pipeline.stats.summary_stats import DEFAULT_TOP_K_CONFIG, GROUP, BaseStats, TopKConfig

class LanguagePredictor:
    @abstractmethod
    def predict(self, doc: Document) -> float:
        raise NotImplemented


class CLDModel(LanguagePredictor):
    def __init__(self) -> None:
        import cld3
        super().__init__()

    def predict(self, doc: Document) -> tuple[str, float]:
        prediction = cld3.get_language(doc.text)
        return prediction.language, prediction.probability

class FastTextModel:
    LANGUAGE_ID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    def __init__(self) -> None:
        super().__init__()
        self._model = None

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
    
    def predict(self, doc: Document):
        language, score = self.model.predict(doc.text.replace("\n", " "))
        # language label is given in the form __label__<language_id>
        language = language[0].split("__")[2]
        return language, score[0]

    def predict_normalized(self, doc: Document):
        language, score = self.predict(doc)
        return language, score / 100


class DocStats(BaseStats, ):
    """
    Summary stats of document level metrics:

    Available stats:
    length: Length of the document
    white_space_ratio: Ratio of whitespace characters
    non_alpha_digit_ratio: Ratio of non-alphabetic and non-digit characters
    digit_ratio: Ratio of digits
    """

    type = "📊 - STATS"
    name = "🎤 Language stats"

    def __init__(
        self,
        output_folder: DataFolderLike,
        histogram_round_digits: int = 3,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__(output_folder, groups_to_compute, histogram_round_digits, top_k_config)

    @property
    def model(self):
        if not self._model:
            from fasttext.FastText import _FastText

            model_file = cached_asset_path_or_download(
                LANGUAGE_ID_MODEL_URL,
                namespace="filters",
                subfolder="language_filter",
                desc="fast-text language identifier model",
            )
            self._model = _FastText(model_file)
        return self._model

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        fasttext_en_score = fasttext.predict(doc.text, k=1)
        return {
            "fasttext_en_score": 
        }