from typing import List

from datatrove.data import Document
from datatrove.pipeline.enrishers.base_enrisher import BaseEnrisher
from datatrove.utils.text import SPLIT_TEXT_DOCUMENTS, split_into_parts


class TransformerClassifierEnrisher(BaseEnrisher):
    """Adds the output of a Transformer classifier to the metadata of the document.

    Args:
        model_name: name of the model to use
        field_name: field name to use for the classification metadata
        split_mode: predict and filter on DOCUMENT, PARAGRAPH or SENTENCE level
        store_units: store the units in the metadata
        kwargs: additional arguments to pass to the TextClassificationPipeline
    """

    name = "🤖 Transformer Enrisher"
    _requires_dependencies = ["transformers"]

    def __init__(
        self,
        model_name_or_path: str,
        field_name: str,
        split_mode: str = SPLIT_TEXT_DOCUMENTS,
        store_units: str = False,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(batch_size)
        self.model_name_or_path = model_name_or_path
        self.field_name = field_name
        self.split_mode = split_mode
        self.store_units = store_units
        self._model = None
        self._kwargs = kwargs

    @property
    def model(self):
        if self._model is None:
            from transformers import pipeline

            self._model = pipeline(
                "text-classification",
                model=self.model_name_or_path,
                batch_size=self.batch_size,
                **self._kwargs,
            )
        return self._model

    def enrish_batch(self, batch: List[Document]) -> List[Document]:
        text_batch = []
        batch_id_to_text_batch_id_map = {}
        for idx, doc in enumerate(batch):
            units = split_into_parts(doc.text, mode=self.split_mode)
            batch_id_to_text_batch_id_map[idx] = [idx + i for i in range(len(units))]
            text_batch.extend(units)

        scores = self.model(text_batch)

        for idx, doc in enumerate(batch):
            label_scores = []
            for text_id in batch_id_to_text_batch_id_map[idx]:
                _label_scores = {}
                _label_scores["score"] = scores[text_id]
                if self.store_units:
                    _label_scores["unit"] = text_batch[text_id]
                label_scores.append(_label_scores)
            doc.metadata[self.field_name] = label_scores

        return batch

    def enrish(self, doc: Document) -> Document:
        units = split_into_parts(doc.text, mode=self.split_mode)

        self.stat_update("doc-total")
        self.stat_update("units", value=len(units), unit="doc")

        label_scores = []
        for unit in units:
            scores = self.model(unit)
            _label_scores = {}
            _label_scores["score"] = scores
            if self.store_units:
                _label_scores["unit"] = unit
            label_scores.append(_label_scores)

        doc.metadata[self.field_name] = label_scores
        return doc
