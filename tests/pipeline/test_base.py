from unittest import TestCase

from datatrove.pipeline.base import PipelineStep
from datatrove.utils._import_utils import _is_distribution_available


class DummyPipelineStep(PipelineStep):
    _requires_dependencies = [
        "numpy",
        "non_existent_dependency1",
        ("non_existent_dependency2", "non_existent_dependency2-wheel"),
    ]


class Pep508InstalledStep(PipelineStep):
    _requires_dependencies = [
        ("numpy", "numpy @ git+https://github.com/numpy/numpy.git"),
    ]

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        return data


class Pep508MissingStep(PipelineStep):
    _requires_dependencies = [
        ("definitely_missing_pkg_xyz", "definitely-missing-pkg-xyz @ git+https://example.com/foo.git"),
    ]


class TestPipelineStep(TestCase):
    def test_init_pipeline_step_with_missing_dependencies(self):
        with self.assertRaisesRegex(
            ImportError,
            "`non_existent_dependency1` and `non_existent_dependency2`.*`pip install non_existent_dependency1 non_existent_dependency2-wheel`",
        ):
            DummyPipelineStep()

    def test_init_pipeline_step_with_pep508_direct_reference(self):
        Pep508InstalledStep()

    def test_missing_pep508_keeps_install_hint(self):
        with self.assertRaisesRegex(
            ImportError,
            r"`pip install definitely-missing-pkg-xyz @ git\+https://example.com/foo.git`",
        ):
            Pep508MissingStep()


class TestDistributionAvailability(TestCase):
    def test_pep508_direct_reference_uses_distribution_name(self):
        self.assertTrue(_is_distribution_available("numpy"))
        self.assertTrue(_is_distribution_available("numpy @ git+https://github.com/numpy/numpy.git"))
        self.assertTrue(_is_distribution_available("numpy>=1.0"))

    def test_missing_distribution(self):
        self.assertFalse(_is_distribution_available("definitely-not-a-real-distribution-xyz"))
