import os
from typing import Tuple

from fsspec.core import strip_protocol
from huggingface_hub import cached_assets_path
from loguru import logger

from datatrove.data import Document
from datatrove.io import download_file
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


class FastTextClassifierFilter(BaseFilter):
    name = "🤖 fastText"
    _requires_dependencies = [("fasttext", "fasttext-wheel")]

    def __init__(
        self,
        model_url: str,
        filter_labels: Tuple[str, float] | list[Tuple[str, float]] = None,
        save_labels_in_metadata: bool = True,
        exclusion_writer: DiskWriter = None,
    ):
        """
        Only keeps documents that have at least one of the labels in `filter_labels` with a score above the configured
        threshold.
        Example:
            for `filter_labels=[("math", 0.9)]` will only keep samples with a score on __label__math of at least 0.9

        Info to train your own classifier: https://fasttext.cc/docs/en/supervised-tutorial.html

        Args:
            model_url: url to download the model from
            filter_labels: tuple of (label name without "__label__", min score) (or list of such tuples)
            save_labels_in_metadata: whether to save all the label scores in the document metadata
            exclusion_writer:
        """
        super().__init__(exclusion_writer)
        self.model_url = model_url
        self.filter_labels = filter_labels
        if filter_labels and isinstance(filter_labels[0], str):
            self.filter_labels = [filter_labels]
        if not self.filter_labels:
            logger.warning("No labels to filter provided. All samples will be kept.")
        self.save_labels_in_metadata = save_labels_in_metadata
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
        labels, scores = self.model.predict(doc.text.replace("\n", ""))
        label_scores = dict(zip(labels, scores))
        if self.save_labels_in_metadata:
            doc.metadata.update(label_scores)
        return not self.filter_labels or any(
            label_scores.get(f"__label__{label}", -9e9) >= min_score for label, min_score in self.filter_labels
        )
