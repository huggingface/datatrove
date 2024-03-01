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
    """
    Only keeps documents that have
    - AT LEAST ONE of the labels in `keep_labels` with a score above the configured threshold, or
    - NONE of the labels in `remove_labels` with a score above the configured threshold.

    You can only supply one of these, to avoid conflicts. Use multiple filters if you need to. If you supply
    neither, the block will simply annotate each document with the labels (set `save_labels_in_metadata=True`)

    Example:
        for `keep_labels=[("math", 0.9)]` will only keep samples with a score on __label__math of at least 0.9
        for `remove_labels=[("math", 0.9)]` will remove samples with a score on __label__math of at least 0.9

    Info to train your own classifier: https://fasttext.cc/docs/en/supervised-tutorial.html

    Args:
        model_url: url to download the model from or local path
        keep_labels: tuple of (label name without "__label__", min score) (or list of such tuples)
        remove_labels: tuple of (label name without "__label__", min score) (or list of such tuples)
        save_labels_in_metadata: whether to save all the label scores in the document metadata
        exclusion_writer:
    """

    name = "🤖 fastText"
    _requires_dependencies = [("fasttext", "fasttext-wheel")]

    def __init__(
        self,
        model_url: str,
        keep_labels: Tuple[str, float] | list[Tuple[str, float]] = None,
        remove_labels: Tuple[str, float] | list[Tuple[str, float]] = None,
        save_labels_in_metadata: bool = True,
        exclusion_writer: DiskWriter = None,
        newline_replacement="",
    ):
        super().__init__(exclusion_writer)
        self.model_url = model_url
        self.keep_labels = keep_labels
        self.remove_labels = remove_labels
        if keep_labels and remove_labels:
            raise ValueError("You can only supply one of `keep_labels` or `remove_labels`.")
        self.newline_replacement = newline_replacement
        if keep_labels and isinstance(keep_labels[0], str):
            self.keep_labels = [keep_labels]
        if remove_labels and isinstance(remove_labels[0], str):
            self.remove_labels = [remove_labels]
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
        labels, scores = self.model.predict(doc.text.replace("\n", self.newline_replacement))
        label_scores = dict(zip(labels, scores))
        if self.save_labels_in_metadata:
            doc.metadata.update(label_scores)
        if self.keep_labels:
            return any(
                label_scores.get(f"__label__{label}", -9e9) >= min_score for label, min_score in self.keep_labels
            )
        else:
            return not self.remove_labels or not any(
                label_scores.get(f"__label__{label}", -9e9) >= min_score for label, min_score in self.remove_labels
            )
