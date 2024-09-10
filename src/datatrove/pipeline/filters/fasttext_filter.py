from collections import defaultdict
from typing import Tuple

import numpy as np

from datatrove.data import Document
from datatrove.io import cached_asset_path_or_download
from datatrove.pipeline.filters.base_filter import PRECALCULATED_STATS, BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.logging import logger
from datatrove.utils.text import SPLIT_TEXT_DOCUMENTS, split_into_parts


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
        newline_replacement: str to replace \n with before predicting scores
        filter_mode: predict and filter on DOCUMENT, PARAGRAPH or SENTENCE level
        exclusion_writer:
    """

    name = "🤖 fastText"
    _requires_dependencies = [("fasttext", "fasttext-wheel"), "fasteners"]

    def __init__(
        self,
        model_url: str,
        precalculated_stats: PRECALCULATED_STATS = PRECALCULATED_STATS.re_calculate_if_missing,
        field_name: str = "fasttext_classifier",
        keep_labels: Tuple[str, float] | list[Tuple[str, float]] | None = None,
        remove_labels: Tuple[str, float] | list[Tuple[str, float]] | None = None,
        save_labels_in_metadata: bool = True,
        exclusion_writer: DiskWriter | None = None,
        newline_replacement="",
        filter_mode: str = SPLIT_TEXT_DOCUMENTS,
    ):
        super().__init__(exclusion_writer)
        self.model_url = model_url
        self.precalculated_stats = precalculated_stats
        self.field_name = field_name
        self.keep_labels = keep_labels
        self.remove_labels = remove_labels
        self.filter_mode = filter_mode
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

    def _filter_from_existing_stats(self, doc: Document) -> bool | tuple[bool, str]:
        def check_label_scores(unit_scores):
            if self.keep_labels:
                return any(
                    unit_scores.get(f"__label__{label}", -9e9) >= min_score for label, min_score in self.keep_labels
                )
            else:
                return not self.remove_labels or not any(
                    unit_scores.get(f"__label__{label}", -9e9) >= min_score for label, min_score in self.remove_labels
                )

        if "units" not in doc.metadata[self.field_name]:
            units = split_into_parts(doc.text, mode=self.split_mode)
        else:
            units = doc.metadata[self.field_name]["units"]

        if "label_scores" not in doc.metadata[self.field_name]:
            logger.warning("No label scores found in metadata. Check that the Enrisher was run before the filter.")
            return False, "no_label_scores"

        len_units = len(units)
        if any(
            len_units != len(doc.metadata[self.field_name]["label_scores"][label])
            for label in doc.metadata[self.field_name]["label_scores"]
        ):
            logger.warning(
                "Number of units in document does not match number of label scores."
                "Check that the split mode is the same as in the Enrisher."
            )
            return False, "split_mode_mismatch"

        kept_spans = []
        label_scores = doc.metadata[self.field_name]["label_scores"]
        for idx, unit in enumerate(units):
            labels = list(label_scores.keys())
            scores = [label_scores[label][idx] for label in labels]
            if check_label_scores(dict(zip(labels, scores))):
                kept_spans.append(unit)
                self.stat_update("kept_span")
            else:
                self.stat_update("removed_span")

        doc.text = "".join(kept_spans)
        return not not doc.text.strip()

    def _filter_maybe_from_existing_stats(self, doc: Document) -> bool | tuple[bool, str]:
        def check_label_scores(unit_scores):
            if self.keep_labels:
                return any(
                    unit_scores.get(f"__label__{label}", -9e9) >= min_score for label, min_score in self.keep_labels
                )
            else:
                return not self.remove_labels or not any(
                    unit_scores.get(f"__label__{label}", -9e9) >= min_score for label, min_score in self.remove_labels
                )

        units = doc.metadata.get(self.field_name, {}).get("units", None)

        if units is None or self.precalculated_stats == PRECALCULATED_STATS.re_calculate:
            units = split_into_parts(doc.text, mode=self.filter_mode)

        _force_recalc = False
        if (
            self.precalculated_stats == PRECALCULATED_STATS.re_calculate_if_missing
            and "label_scores" not in doc.metadata.get(self.field_name, {})
        ):
            _force_recalc = True

        if self.precalculated_stats == PRECALCULATED_STATS.re_calculate:
            _force_recalc = True

        kept_spans = []
        label_scores = (
            defaultdict(list) if _force_recalc else doc.metadata.get(self.field_name, {}).get("label_scores")
        )
        for idx, unit in enumerate(units):
            if _force_recalc:
                labels, scores = self.model.predict(unit.strip().replace("\n", self.newline_replacement), k=-1)
            else:
                labels = list(label_scores.keys())
                scores = [label_scores[label][idx] for label in labels]
            if self.save_labels_in_metadata:
                for label, score in zip(labels, scores):
                    label_scores[label].append(score)
            if check_label_scores(dict(zip(labels, scores))):
                kept_spans.append(unit)
                self.stat_update("kept_span")
            else:
                self.stat_update("removed_span")

        doc.text = "".join(kept_spans)
        if self.save_labels_in_metadata and _force_recalc:
            doc.metadata.update(
                {self.field_name: {label: np.mean(scores).item() for label, scores in label_scores.items()}}
            )
        return not not doc.text.strip()

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        if (
            self.precalculated_stats == PRECALCULATED_STATS.re_calculate
            or self.precalculated_stats == PRECALCULATED_STATS.re_calculate_if_missing
        ):
            return self._filter_maybe_from_existing_stats(doc)
        elif self.precalculated_stats == PRECALCULATED_STATS.re_use:
            if self.field_name not in doc.metadata:
                logger.warning(f"{self.field_name} not found in metadata.")
                return False, "missing_fasttext_field"
            return self._filter_from_existing_stats(doc)
        else:
            raise ValueError(f"Unknown precalculated_stats: {self.precalculated_stats}")
