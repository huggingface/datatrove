from abc import abstractmethod
from typing import get_args

from datatrove.data import Document
from datatrove.io import DataFolderLike, cached_asset_path_or_download
from datatrove.pipeline.stats.summary_stats import DEFAULT_TOP_K_CONFIG, GROUP, BaseStats, TopKConfig

class LanguagePredictor:
    def __init__(self, language: str) -> None:
        self.language = language

    @abstractmethod
    def predict(self, doc: Document) -> float:
        raise NotImplemented


class CLDModel(LanguagePredictor):
    def __init__(self, language: str) -> None:
        super().__init__(language)

    def predict(self, doc: Document) -> float:
        import cld3
        prediction = cld3.get_frequent_language(doc.text, 10)
        lang_id = [x.language for x in prediction].index(self.language)
        if lang_id == -1:
            return 0.0
        return prediction[lang_id].probability


class FastTextModel(LanguagePredictor):
    LANGUAGE_ID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    def __init__(self, language: str) -> None:
        super().__init__(language)
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
        langs, score = self.model.predict(doc.text.replace("\n", " "), k=10)
        lang_id = [lang.split("__")[2] for lang in langs].index(self.language)
        if lang_id == -1:
            return 0.0
        return score[lang_id]

class LangStats(BaseStats):
    """
    Summary stats of document level metrics:

    Available stats:
    cld3_{language}
    fasttext_{language}
    """

    type = "📊 - STATS"
    name = "🎤 Language stats"

    def __init__(
        self,
        output_folder: DataFolderLike,
        language: str,
        histogram_round_digits: int = 3,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__(output_folder, groups_to_compute, histogram_round_digits, top_k_config)
        self.fasttext = FastTextModel(language)
        self.cld3 = CLDModel(language)
        self.language = language

    
    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        return {
            f"cld3_{self.language}": self.cld3.predict(doc),
            f"fasttext_{self.language}": self.fasttext.predict(doc),
        }