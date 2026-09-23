# tests/unit/experiment/test_survey_script_view.py

# F811: ``survey_script`` is a pytest fixture looked up by name.
# ruff: noqa: N803, S101, F811

"""``--os-or-ts`` resolution: ledger default, explicit override, and the gates.

Protocol §2.3 derives the training view from the method's native sampling
assumption, so the flag overrides a default rather than introducing a free
parameter.  A calibrated request is refused wherever it would have no meaning:
for a method the ledger declares native to OS sampling, and for the PN oracle,
which trains on real labels and generates no PU label view at all.
"""

import pytest
from _survey_script_helpers import make_splits, survey_script  # noqa: F401 - pytest fixture

pytestmark = pytest.mark.unit


@pytest.fixture
def ledger(survey_script):
    return survey_script.load_ledger(survey_script.LEDGER_PATH)


class _ViewAwareEstimator:
    """Stand-in for an estimator whose ``fit`` carries the view hook."""

    def fit(self, X, y, *, class_prior=None, os_or_ts=None):  # noqa: N803
        return self


class _PlainEstimator:
    """Stand-in for an estimator whose ``fit`` has no view hook."""

    def fit(self, X, y, *, class_prior=None):  # noqa: N803
        return self


class TestResolution:
    def test_default_follows_the_ledger_native_assumption(self, survey_script, ledger):
        """Both §2.3 conditions: native TS *and* an interface that can carry it."""
        assert (
            survey_script.resolve_training_view(
                ledger, "nnpu", None, is_oracle=False, estimator_class=_ViewAwareEstimator
            )
            == "ts"
        )
        assert (
            survey_script.resolve_training_view(
                ledger, "kldce", None, is_oracle=False, estimator_class=_ViewAwareEstimator
            )
            == "os"
        )

    def test_native_ts_without_the_interface_stays_on_os(self, survey_script, ledger):
        """An unwired native-TS method must still run, not fail every run."""
        assert (
            survey_script.resolve_training_view(
                ledger, "nnpu", None, is_oracle=False, estimator_class=_PlainEstimator
            )
            == "os"
        )

    def test_explicit_os_overrides_a_native_ts_method(self, survey_script, ledger):
        """The ablation direction must stay available."""
        assert (
            survey_script.resolve_training_view(
                ledger, "nnpu", "os", is_oracle=False, estimator_class=_ViewAwareEstimator
            )
            == "os"
        )

    def test_explicit_ts_on_a_native_ts_method_is_kept(self, survey_script, ledger):
        assert (
            survey_script.resolve_training_view(
                ledger, "nnpu", "ts", is_oracle=False, estimator_class=_ViewAwareEstimator
            )
            == "ts"
        )

    def test_ts_is_refused_for_a_native_os_method(self, survey_script, ledger):
        with pytest.raises(ValueError, match="declared native to 'os'"):
            survey_script.resolve_training_view(
                ledger, "kldce", "ts", is_oracle=False, estimator_class=_ViewAwareEstimator
            )

    def test_explicit_ts_without_the_interface_names_the_gap(self, survey_script, ledger):
        with pytest.raises(ValueError, match="declares no os_or_ts parameter"):
            survey_script.resolve_training_view(
                ledger, "nnpu", "ts", is_oracle=False, estimator_class=_PlainEstimator
            )

    def test_oracle_refuses_a_calibrated_view(self, survey_script, ledger):
        with pytest.raises(ValueError, match="does not apply"):
            survey_script.resolve_training_view(ledger, "pn_oracle", "ts", is_oracle=True)

    def test_oracle_defaults_to_the_os_view(self, survey_script, ledger):
        assert (
            survey_script.resolve_training_view(ledger, "pn_oracle", None, is_oracle=True) == "os"
        )


class TestCliGate:
    def test_cli_refuses_ts_for_a_native_os_method(self, survey_script, tmp_path, capsys):
        data_dir = tmp_path / "splits"
        data_dir.mkdir()
        make_splits(data_dir)
        rc = survey_script.main(
            [str(data_dir), "--method", "kldce", "--os-or-ts", "ts", "--seeds", "0"]
        )
        assert rc == 1
        assert "does not apply" in capsys.readouterr().err

    def test_cli_refuses_ts_for_the_oracle(self, survey_script, tmp_path, capsys):
        data_dir = tmp_path / "splits"
        data_dir.mkdir()
        make_splits(data_dir)
        rc = survey_script.main([str(data_dir), "--oracle", "--os-or-ts", "ts", "--seeds", "0"])
        assert rc == 1
        assert "does not apply" in capsys.readouterr().err
