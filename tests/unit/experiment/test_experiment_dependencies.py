"""Public experiments and result consumers work without Survey process modules."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("consumer", ["runner", "results", "reporting"])
def test_basic_param_determ_public_consumers_do_not_import_survey_process_modules(consumer):
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class NoSurveyProcess(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                name = fullname.rsplit(".", 1)[-1]
                if fullname.startswith("pu_toolbox.experiment.survey_") and name in {
                    "survey_protocol", "survey_comparison", "survey_profile_map", "survey_audit",
                    "survey_recipe_binding", "survey_recipe_registry"
                } or fullname == "pu_toolbox.experiment.method_ledger":
                    raise AssertionError("public consumer imported " + fullname)

        sys.meta_path.insert(0, NoSurveyProcess())
        consumer = sys.argv[1]
        if consumer == "runner":
            import numpy as np
            from pu_toolbox.estimators.risk.upu import UPUClassifier
            from pu_toolbox.experiment import DatasetPart, ExperimentRunner

            x = np.random.RandomState(3).normal(size=(50, 3))
            y = np.tile([0, 1], 25)
            parts = [DatasetPart(X=x[start:end], labels=y[start:end], view="clean",
                                 indices=np.arange(start, end), for_selection=(start != 40))
                     for start, end in ((0, 20), (20, 30), (30, 40), (40, 50))]
            def run():
                return ExperimentRunner(seed=3, class_prior=0.5, config={"c": 0.5}).fit(
                    UPUClassifier(0.5, random_state=3), *parts)
            first, second = run(), run()
            assert set(first.test_metrics) == {"PA", "OA"}
            assert first.test_metrics == second.test_metrics
        elif consumer == "results":
            from pu_toolbox.experiment.results import build_summary
            assert build_summary([])["coverage"]["rows"] == 0
        else:
            from pu_toolbox.experiment.reporting import export, write_summary
            assert callable(export) and callable(write_summary)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, consumer],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_edge_unknown_public_attribute_is_not_silently_exported():
    import pu_toolbox.experiment as experiment

    with pytest.raises(AttributeError, match="no attribute"):
        _ = experiment.not_an_experiment_api
    assert "ExperimentRunner" in dir(experiment)
