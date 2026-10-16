"""Plan and input guards for independent, exploratory LBE comparisons."""

import json

import pytest
from _lbe_comparison_helpers import comparison, plan_for, write_split

pytestmark = pytest.mark.unit


def test_basic_plan_is_paired_nonformal_and_preserves_default_budgets(tmp_path):
    root = write_split(tmp_path)
    plan = comparison.build_plan(root, dataset="spambase", seeds=[0], c_tokens=["0.5"])
    assert plan["formal_admission"] is False and plan["budget_matched"] is False
    assert plan["selection_scope"] == "terminal" and not plan["test_used_for_selection"]
    params = plan["pairs"][0]["parameters"]
    assert params["lbe"]["n_em_iter"] == 20
    assert params["lbe_neural"]["n_em_iter"] == 20
    assert params["lbe_neural"]["m_steps"] == 40
    assert params["lbe_neural"]["pretrain_steps"] == 20


def test_determ_planning_is_repeatable_and_does_not_rewrite_inputs(tmp_path):
    root = write_split(tmp_path)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert plan_for(root) == plan_for(root)
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("change", ["seed", "prior", "size", "indices", "dataset"])
def test_param_bad_split_metadata_is_rejected(tmp_path, change):
    root = write_split(tmp_path)
    path = root / "split_0/split_manifest.json"
    metadata = json.loads(path.read_text())
    if change == "seed":
        metadata["seed"] = 1
    elif change == "prior":
        metadata["class_prior"]["population"] = True
    elif change == "size":
        metadata["role_sizes"]["train"] += 1
    elif change == "indices":
        metadata["indices"]["train"][0] += 999
    else:
        metadata["dataset"] = "imdb"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        plan_for(root)


@pytest.mark.parametrize(
    "seeds,tokens",
    [([], ["0.1"]), ([0, 0], ["0.1"]), ([5], ["0.1"]), ([0], ["0.10"]), ([0], ["0.5", "0.5"])],
)
def test_param_invalid_axes_are_rejected_before_loading_splits(tmp_path, seeds, tokens):
    with pytest.raises(ValueError):
        comparison.build_plan(tmp_path / "absent", dataset="spambase", seeds=seeds, c_tokens=tokens)


def test_edge_requested_seed_is_not_the_same_split_with_a_different_label(tmp_path):
    root = write_split(tmp_path, seed=0)
    write_split(tmp_path, seed=1)
    plan = comparison.build_plan(root, dataset="spambase", seeds=[0, 1], c_tokens=["0.5"])
    assert [pair["parameters"]["lbe_neural"]["random_state"] for pair in plan["pairs"]] == [0, 1]
    assert (
        plan["pairs"][0]["split_ref"]["files_sha256"]
        != plan["pairs"][1]["split_ref"]["files_sha256"]
    )


def test_edge_missing_original_file_is_not_silently_regenerated(tmp_path):
    root = write_split(tmp_path)
    (root / "split_0/test.npz").unlink()
    with pytest.raises(FileNotFoundError):
        plan_for(root)


def test_basic_cli_dry_run_neither_fits_nor_creates_output(tmp_path, capsys, monkeypatch):
    pytest.importorskip("torch")
    root = write_split(tmp_path)
    output = tmp_path / "unused"

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not fit")

    monkeypatch.setattr(comparison, "run_pair", forbidden)
    assert (
        comparison.main(
            [str(root), "--out-dir", str(output), "--seeds", "0", "--c", "0.5", "--dry-run"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["formal_admission"] is False
    assert not output.exists()
