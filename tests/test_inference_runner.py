import random
import json
import pytest

from datatrove.data import Document
from datatrove.pipeline.inference.run_inference import (
    InferenceRunner,
    InferenceConfig,
)
from datatrove.pipeline.base import PipelineStep


class CollectorStep(PipelineStep):
    """Collect documents that flow through this step for assertions."""

    def __init__(self):
        super().__init__()
        self.collected: list[Document] = []

    def run(self, data, rank: int = 0, world_size: int = 1):
        for doc in data:
            self.collected.append(doc)
            # propagate so the pipeline can keep going if needed
            yield doc


@pytest.mark.timeout(30)
def test_inference_runner_dummy_end_to_end(tmp_path):
    # Prepare two dummy documents
    docs = [
        Document(text="", id="doc1"),
        Document(text="", id="doc2"),
    ]

    # Simple query builder that wraps the doc id
    def query_builder(doc: Document):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Process {doc.id}"},
                    ],
                }
            ]
        }

    # Collect results after inference
    collector = CollectorStep()

    # Use a random high port to avoid collisions when running tests in parallel
    port = random.randint(35000, 45000)

    config = InferenceConfig(server_port=port, server_type="dummy")

    runner = InferenceRunner(
        records_reader=[],  # not used anymore
        query_builder=query_builder,
        config=config,
        post_process_steps=[collector],
    )

    # Run the pipeline (synchronously)
    runner.run(iter(docs))

    # Assertions: collector captured as many docs as were supplied
    assert len(collector.collected) == len(docs)
    for d in collector.collected:
        # The dummy server returns JSON string in d.text
        payload = json.loads(d.text)
        assert payload["natural_text"].startswith("This is dummy text")
        # finish_reason usage metadata should exist
        assert "usage" in d.metadata 