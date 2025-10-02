"""Post-processing steps for InferenceRunner results."""

from typing import Iterable

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.run_inference import InferenceSuccess


class ExtractInferenceText(PipelineStep):
    """Extract text from InferenceRunner results.

    InferenceRunner stores results in document.metadata["inference_results"] as a list of
    InferenceSuccess/InferenceFailure objects. This step extracts the text from successful
    results and sets it as the document's main text field.

    For vision models (e.g., RolmOCR), each page generates one InferenceSuccess result.
    This step concatenates all page texts with newlines.

    Args:
        remove_inference_results: If True, removes the inference_results metadata field
            after extraction to save memory. Default: True.
    """

    def __init__(self, remove_inference_results: bool = True):
        super().__init__()
        self.remove_inference_results = remove_inference_results

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        for document in data:
            # Extract text from inference results
            inference_results = document.metadata.get("inference_results", [])

            # Concatenate successful results, include error messages for failures
            document.text = "\n".join([
                x.text if isinstance(x, InferenceSuccess) else x.error
                for x in inference_results
            ])

            # Optionally clean up inference_results metadata
            if self.remove_inference_results and "inference_results" in document.metadata:
                del document.metadata["inference_results"]

            self.stat_update("documents_processed")
            yield document
