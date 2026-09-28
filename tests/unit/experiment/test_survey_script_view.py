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

import json

import pytest
from _survey_script_helpers import (  # noqa: F401 - pytest fixture
    make_sar_splits,
    make_splits,
    survey_script,
)

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

    def test_explicit_ts_without_an_estimator_class_is_refused(self, survey_script, ledger):
        """No class means the interface was never checked, so ``ts`` cannot be claimed.

        ``--os-or-ts ts`` is the operator stating which view the run must take.
        A caller that omits the estimator class leaves the second §2.3 condition
        unverifiable, and answering ``ts`` anyway would hand back an unverified
        promise.  Falling back to ``os`` is no better: that is exactly the silent
        view downgrade this resolution exists to prevent.
        """
        with pytest.raises(ValueError, match="no estimator class was supplied"):
            survey_script.resolve_training_view(ledger, "nnpu", "ts", is_oracle=False)

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


class TestViewScopedPaths:
    """A run directory is keyed by its view: the manifest name is not unique.

    Both views write ``manifest.json``, and ``method_ledger_entry.json`` beside
    it, so two views sharing a directory means the later run silently replaces
    the earlier one's record while its checkpoints stay behind.
    """

    def test_run_directory_differs_by_view(self, survey_script, tmp_path):
        c_value = survey_script.CValue(value=0.1, token="0.1")
        dirs = {
            view: survey_script._run_directory(
                tmp_path,
                mechanism="scar",
                c_value=c_value,
                seed=0,
                is_oracle=False,
                run_view=view,
            )
            for view in ("os", "ts")
        }
        assert dirs["os"] != dirs["ts"]
        # The view is the outermost layer, so the c/seed tail stays readable.
        for view, path in dirs.items():
            assert path.parent.name == "c_0.1"
            assert path.name == "seed_0"
            assert path.parent.parent.name == view

    def test_sar_mechanism_keeps_its_layer_under_the_view(self, survey_script, tmp_path):
        """mechanism and view compose without either one shadowing the other."""
        path = survey_script._run_directory(
            tmp_path,
            mechanism="sar_lbe_a",
            c_value=survey_script.CValue(value=0.05, token="0.05"),
            seed=2,
            is_oracle=False,
            run_view="os",
        )
        assert path.parent.parent.name == "sar_lbe_a"
        assert path.parent.parent.parent.name == "os"

    def test_oracle_carries_the_view_layer_too(self, survey_script, tmp_path):
        """The oracle has no view choice, but shares the layout so nothing collides."""
        path = survey_script._run_directory(
            tmp_path, mechanism="scar", c_value=None, seed=1, is_oracle=True, run_view="os"
        )
        assert path.parent.name == "c_independent"
        assert path.name == "seed_1"
        assert path.parent.parent.name == "os"

    def test_second_view_does_not_overwrite_the_first(self, survey_script, tmp_path):
        """End-to-end: one --out-dir must keep both runs' records."""
        pytest.importorskip("torch", reason="nnpu needs PyTorch for the ts hook")
        data_dir = tmp_path / "splits"
        data_dir.mkdir()
        # nnpu floors at 2 labeled positives in train *and* pu_val, so the roster
        # needs more positives than ``make_splits`` carries at c=0.5.
        make_sar_splits(data_dir)
        # that helper writes only the npz bundle; the script reads the prior and
        # the role sizes back from the manifest beside it.
        (data_dir / "split_manifest.json").write_text(
            json.dumps(
                {
                    "dataset": "spambase",
                    "seed": 0,
                    "role_sizes": {"train": 60, "pu_val": 20, "clean_val": 20, "test": 20},
                    "class_prior": {"population": 0.5},
                }
            ),
            encoding="utf-8",
        )
        out_dir = tmp_path / "out"
        common = [
            str(data_dir),
            "--method",
            "nnpu",
            "--dataset",
            "spambase",
            "--protocol",
            "survey-v1.2",
            "--seeds",
            "0",
            "--c",
            "0.5",
            "--class-prior",
            "0.5",
        ]
        assert survey_script.main(common + ["--out-dir", str(out_dir), "--os-or-ts", "ts"]) == 0
        assert survey_script.main(common + ["--out-dir", str(out_dir), "--os-or-ts", "os"]) == 0
        manifests = sorted(out_dir.rglob("manifest.json"))
        assert len(manifests) == 2
        views = {json.loads(path.read_text(encoding="utf-8"))["run_view"] for path in manifests}
        assert views == {"ts-compatible", "os-compatible"}
