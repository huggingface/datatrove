from unittest import TestCase

from datatrove.pipeline.base import PipelineStep


class DummyPipelineStep(PipelineStep):
    _requires_dependencies = [
        "numpy",
        "non_existent_dependency1",
        ("non_existent_dependency2", "non_existent_dependency2-wheel"),
    ]


class TestPipelineStep(TestCase):
    def test_init_pipeline_step_with_missing_dependencies(self):
        with self.assertRaisesRegex(
            ImportError,
            "`non_existent_dependency1` and `non_existent_dependency2`.*`pip install non_existent_dependency1 non_existent_dependency2-wheel`",
        ):
            DummyPipelineStep()
