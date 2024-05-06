from abc import abstractmethod
from typing import get_args

from loguru import logger

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
        self._model = None
    
    @property
    def model(self):
        if not self._model:
            import gcld3
            self._model = gcld3.NNetLanguageIdentifier(0, 10_000)
        return self._model

    def predict(self, doc: Document) -> float:
        prediction = self.model.FindTopNMostFreqLangs(doc.text, 10)
        try:
            lang_id = [x.language for x in prediction].index(self.language)
        except:
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
        try:
            lang_id = [lang.split("__")[2] for lang in langs].index(self.language)
            return score[lang_id]
        except:
            return 0.0

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
        fast_text_score = doc.metadata.get("language_score") if doc.metadata.get("language") == self.language else None
        cld3_score = self.cld3.predict(doc)
        if fast_text_score is None:
            fast_text_score = self.fasttext.predict(doc)
        data = {
            f"cld3_{self.language}": cld3_score,
            f"fasttext_{self.language}": fast_text_score,
            f"cld3_sub_fasttext_{self.language}": cld3_score - fast_text_score,
        }
        return data
