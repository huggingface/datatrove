from collections import defaultdict

import numpy as np

from datatrove.data import Document
from datatrove.io import cached_asset_path_or_download
from datatrove.pipeline.enrichers.base_enricher import BaseEnricher
from datatrove.utils.text import SPLIT_TEXT_DOCUMENTS, split_into_parts


class FastTextClassifierEnricher(BaseEnricher):
    """Adds the output of a FastText classifier to the metadata of the document.

    We keep the spans o

    Info to train your own classifier: https://fasttext.cc/docs/en/supervised-tutorial.html

    Args:
        model_url: url to download the model from or local path
        newline_replacement: str to replace \n with before predicting scores
        split_mode: predict and filter on DOCUMENT, PARAGRAPH or SENTENCE level
        store_units: store the units in the metadata
    """

    name = "🤖 fastText Enricher"
    _requires_dependencies = [("fasttext", "fasttext-wheel"), "fasteners"]

    def __init__(
        self,
        model_url: str,
        field_name: str = "fasttext_classifier",
        newline_replacement="",
        split_mode: str = SPLIT_TEXT_DOCUMENTS,
        store_units: bool = False,
    ):
        super().__init__()
        self.model_url = model_url
        self.field_name = field_name
        self.split_mode = split_mode
        self.newline_replacement = newline_replacement
        self.store_units = store_units
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from fasttext.FastText import _FastText

            model_file = cached_asset_path_or_download(
                self.model_url,
                namespace="filters",
                subfolder="fasttext",
                desc="fast-text model",
            )
            self._model = _FastText(model_file)
            # check label values
            available_labels = [x.removeprefix("__label__") for x in self._model.labels]
            for label, _ in self.keep_labels or [] + self.remove_labels or []:
                if label not in available_labels:
                    raise ValueError(
                        f"Label '{label}' passed as keep_labels or remove_labels is not available in this "
                        f"FastText model. Available labels: {available_labels}"
                    )
        return self._model

    def enrich(self, doc: Document) -> bool:
        units = split_into_parts(doc.text, mode=self.split_mode)

        self.stat_update("doc-total")
        self.stat_update("units", value=len(units), unit="doc")

        label_scores = defaultdict(list)
        for unit in units:
            labels, scores = self.model.predict(unit.strip().replace("\n", self.newline_replacement), k=-1)
            for label, score in zip(labels, scores):
                label_scores[label].append(score)

        doc.metadata[self.field_name] = {
            "label_scores": label_scores,
            "label_scores_mean": {label: np.mean(scores) for label, scores in label_scores.items()},
        }
        if self.store_units:
            doc.metadata[self.field_name]["units"] = units

        return doc
