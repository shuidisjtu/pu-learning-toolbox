"""Real archive/index verification is never narrowed by audit selection."""

import importlib
import json

import pytest
from _review_split_receipt_helpers import make_receipt

pytestmark = pytest.mark.unit


def reviewer():
    return importlib.import_module("scripts.review_survey_split_receipt")


def test_basic_determ_receipt_report_and_selected_nonzero_seed(tmp_path, capsys):
    root, index, archives = make_receipt(tmp_path, ("adni", "imdb"), (2, 7))
    module = reviewer()
    first = module.audit_receipt(root, index, archives, datasets=("adni",), seeds=(7,))
    second = module.audit_receipt(root, index, archives, datasets=("adni",), seeds=(7,))
    assert first == second
    assert first["integrity"]["ok"]
    assert [(r["dataset"], r["seed"]) for r in first["splits"]] == [("adni", 7)]
    assert first["test_file_hash_count"] == {"adni": 1}
    assert (
        module.main(
            [
                "--splits",
                str(root),
                "--index",
                str(index),
                "--archive-dir",
                str(archives),
                "--datasets",
                "adni",
                "--seeds",
                "7",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == first


@pytest.mark.parametrize("defect", ["unselected_file", "unselected_archive", "extra_file"])
def test_param_edge_full_index_verified_even_for_subset(tmp_path, defect):
    root, index, archives = make_receipt(tmp_path, ("adni", "imdb"))
    if defect == "unselected_file":
        (root / "imdb/split_0/train.npz").write_bytes(b"broken")
    elif defect == "unselected_archive":
        (archives / "imdb-splits.tar").write_bytes(b"broken")
    else:
        (root / "imdb/split_0/unindexed.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(SystemExit, match="Receipt integrity check failed"):
        reviewer().audit_receipt(root, index, archives, datasets=("adni",), seeds=(0,))


@pytest.mark.parametrize(
    "args",
    [
        ("--datasets", "unknown"),
        ("--datasets", ""),
        ("--seeds", ""),
        ("--seeds", "bad"),
        ("--seeds", "-1"),
        ("--seeds", "1,1"),
        ("--datasets", "imdb,imdb"),
    ],
)
def test_param_edge_invalid_selection_rejected_before_receipt(tmp_path, args, capsys):
    with pytest.raises(SystemExit) as error:
        reviewer().main(["--archive-dir", str(tmp_path), *args])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_basic_defaults_keep_pilot_datasets_and_five_seeds(tmp_path):
    root, index, archives = make_receipt(tmp_path, ("spambase", "imdb", "cifar10"), range(5))
    report = reviewer().audit_receipt(root, index, archives)
    assert [(s["dataset"], s["seed"]) for s in report["splits"]] == [
        (dataset, seed) for dataset in ("spambase", "imdb", "cifar10") for seed in range(5)
    ]


@pytest.mark.parametrize("datasets,seeds", [(("imdb",), (0,)), (("adni",), (9,))])
def test_param_edge_selection_not_in_index_fails_closed(tmp_path, datasets, seeds):
    root, index, archives = make_receipt(tmp_path, ("adni",))
    with pytest.raises(ValueError, match="not in receipt index"):
        reviewer().audit_receipt(root, index, archives, datasets=datasets, seeds=seeds)
