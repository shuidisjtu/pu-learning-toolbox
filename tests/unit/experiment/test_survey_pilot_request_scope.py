# tests/unit/experiment/test_survey_pilot_request_scope.py

# ruff: noqa: N803, N806, F811, S101

"""What a request names decides what is read, planned and executed.

Three ways the request and the work could disagree, and the tests that hold
them together.  A ``--protocol`` path reached the runs but not the plan, so the
driver sized and resumed one matrix and trained another.  The splits of a
dataset the request never named were read anyway, so a defect in it could stop
a shard before its first batch.  And a snapshot path that could not be written
cost a traceback rather than an error.
"""

import json

import pytest
from _survey_pilot_helpers import (  # noqa: F401 - imported fixtures are used by name
    ALL_PRIORS,
    driver,
    splits_tree,
    unit_calls,
)

from pu_toolbox.experiment.pilot_plan import population_priors, split_digests
from pu_toolbox.experiment.survey_protocol import PROTOCOL_PATH, resolve_protocol_path

pytestmark = pytest.mark.unit


def _protocol_without(tmp_path, *datasets: str) -> str:
    """A copy of the shipped matrix with some datasets taken out.

    Written to a path, because the point is that the matrix a request is
    *planned* from is the one it hands to its runs: a copy is the only way to
    tell a plan that followed ``--protocol`` from one that ignored it.
    """
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["execution_units"] = [
        unit for unit in payload["execution_units"] if unit["dataset"] not in datasets
    ]
    path = tmp_path / "custom_protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _split(
    root, dataset: str, seed: int, *, prior: float, digest: str, declares: str | None = None
) -> None:
    """One split manifest under *dataset*'s directory, declaring *declares*.

    ``declares`` writes a manifest that names a dataset other than the one whose
    directory holds it, which is how a stray file is spelled.
    """
    path = root / dataset / f"split_{seed}" / "split_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": declares or dataset,
                "seed": seed,
                "indices_sha256": digest,
                "class_prior": {"population": prior},
            }
        ),
        encoding="utf-8",
    )


def _disagreeing_tree(tmp_path, conflicting: str = "imdb"):
    """Spambase fine; *conflicting*'s seeds contradict each other about the prior."""
    root = splits_tree(tmp_path, prior=0.39, datasets=("spambase",))
    _split(root, conflicting, 0, prior=0.5, digest="b" * 64)
    _split(root, conflicting, 1, prior=0.6, digest="b" * 64)
    return root


# --- the protocol a request names is the one it is planned and run from -------


def test_basic_the_plan_comes_from_the_protocol_the_request_names(driver, tmp_path, capsys):
    custom = _protocol_without(tmp_path, "cifar10")
    code = driver.main(
        [
            "--dry-run",
            "--results",
            str(tmp_path / "none"),
            "--splits",
            str(splits_tree(tmp_path, datasets=("imdb", "spambase"))),
            "--protocol",
            custom,
            *ALL_PRIORS,
        ]
    )

    assert code == 0
    assert "planned: 430 run(s)" in capsys.readouterr().out


def test_basic_the_batches_run_under_the_protocol_the_request_names(driver, unit_calls, tmp_path):
    custom = _protocol_without(tmp_path, "cifar10")
    driver.main(
        [
            "--results",
            str(tmp_path / "out"),
            "--splits",
            str(splits_tree(tmp_path, datasets=("imdb", "spambase"))),
            "--protocol",
            custom,
            "--class-prior",
            "imdb=0.5,spambase=0.39",
        ]
    )

    assert len(unit_calls) == 38  # imdb + spambase batches, not the matrix's 57
    assert all(call[call.index("--protocol") + 1] == custom for call in unit_calls)


def test_edge_a_subset_is_validated_against_the_protocol_the_request_names(
    driver, unit_calls, tmp_path, capsys
):
    """The name is checked against the matrix the request names, not the shipped one."""
    custom = _protocol_without(tmp_path, "cifar10")
    code = driver.main(
        [
            "--results",
            str(tmp_path / "out"),
            "--splits",
            str(splits_tree(tmp_path)),
            "--protocol",
            custom,
            "--datasets",
            "cifar10",
            *ALL_PRIORS,
        ]
    )

    assert code == 1
    assert unit_calls == []
    err = capsys.readouterr().err
    assert "cifar10" in err  # the name asked for
    assert "imdb" in err  # what that matrix does plan instead


def test_param_a_protocol_name_resolves_to_the_matrix_it_denotes(tmp_path):
    for alias in ("survey-v1", "survey-v1.1", "survey-v1.2"):
        assert resolve_protocol_path(alias) == PROTOCOL_PATH
    custom = tmp_path / "custom.json"
    assert resolve_protocol_path(str(custom)) == custom.resolve()


def test_edge_a_custom_protocol_scopes_the_split_read_without_the_flag(driver, tmp_path):
    """The matrix decides which splits are read, not the flag that narrowed it.

    A request that named its matrix by path used to read every dataset's splits
    anyway, so a conflict inside one the matrix had dropped could stop it.
    """
    custom = _protocol_without(tmp_path, "imdb")
    code = driver.main(
        [
            "--dry-run",
            "--results",
            str(tmp_path / "none"),
            "--splits",
            str(_disagreeing_tree(tmp_path, conflicting="imdb")),
            "--protocol",
            custom,
            *ALL_PRIORS,
        ]
    )

    assert code == 0


# --- a shard reads its own splits, and only its own ---------------------------


def test_edge_an_unrelated_dataset_cannot_block_the_shard(driver, tmp_path):
    """imdb's defect is not spambase's to report: no run of this shard reads it."""
    code = driver.main(
        [
            "--dry-run",
            "--results",
            str(tmp_path / "none"),
            "--splits",
            str(_disagreeing_tree(tmp_path)),
            "--datasets",
            "spambase",
            *ALL_PRIORS,
        ]
    )

    assert code == 0


def test_basic_the_prior_check_still_covers_every_dataset_of_the_request(driver, tmp_path, capsys):
    """Narrowed to the request, not dropped: the whole matrix still refuses it."""
    code = driver.main(
        [
            "--dry-run",
            "--results",
            str(tmp_path / "none"),
            "--splits",
            str(_disagreeing_tree(tmp_path)),
            *ALL_PRIORS,
        ]
    )

    assert code == 1
    assert "declares two population priors" in capsys.readouterr().err


def test_edge_a_split_read_can_be_limited_to_the_datasets_asked_for(tmp_path):
    root = _disagreeing_tree(tmp_path)

    with pytest.raises(ValueError, match="declares two population priors"):
        population_priors(root)
    assert population_priors(root, datasets=("spambase",)) == {"spambase": 0.39}
    assert split_digests(root) == {
        ("spambase", 0): "a" * 64,
        ("imdb", 0): "b" * 64,
        ("imdb", 1): "b" * 64,
    }
    assert split_digests(root, datasets=("spambase",)) == {("spambase", 0): "a" * 64}


def test_edge_a_split_read_visits_only_the_named_dataset_directories(tmp_path):
    """A manifest sitting in another dataset's directory is not this shard's to read.

    The directory names the dataset, so an allowlist has to scope the walk and
    not only the result: a stray file declaring ``spambase`` while living under
    ``imdb/`` would otherwise contaminate a shard that never asked for imdb.
    """
    root = splits_tree(tmp_path, prior=0.39, datasets=("spambase",))
    _split(root, "imdb", 9, prior=0.5, digest="c" * 64, declares="spambase")

    assert split_digests(root, datasets=("spambase",)) == {("spambase", 0): "a" * 64}
    # With no allowlist the whole root is in scope, and a record is placed by
    # what it declares rather than by where it sits -- the stray is visible
    # here, and a scoped read that never walked imdb/ cannot be told about it.
    assert split_digests(root) == {("spambase", 0): "a" * 64, ("spambase", 9): "c" * 64}


def test_edge_a_bare_string_is_not_a_dataset_list(tmp_path):
    """``"spambase"`` is eight characters, and a set of them reads nothing."""
    root = _disagreeing_tree(tmp_path)

    with pytest.raises(ValueError, match="collection of names"):
        split_digests(root, datasets="spambase")
    with pytest.raises(ValueError, match="collection of names"):
        population_priors(root, datasets="spambase")


def test_basic_an_empty_dataset_list_reads_nothing(tmp_path):
    """Naming no dataset asks for nothing; it does not fall back to everything."""
    root = _disagreeing_tree(tmp_path)

    assert split_digests(root, datasets=()) == {}
    assert population_priors(root, datasets=()) == {}


# --- a snapshot the request cannot write is an error, not a traceback ---------


def test_edge_an_unwritable_snapshot_path_is_refused(driver, tmp_path, capsys):
    path = tmp_path / "absent" / "plan.json"
    code = driver.main(
        [
            "--dry-run",
            "--plan-json",
            str(path),
            "--results",
            str(tmp_path / "none"),
            "--splits",
            str(splits_tree(tmp_path)),
            *ALL_PRIORS,
        ]
    )

    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "plan.json" in err


def test_determ_one_request_writes_the_same_snapshot_every_time(driver, tmp_path):
    """The snapshot is what two hosts compare, so equal requests owe equal bytes.

    Sorted dataset names and a plan in protocol order are what make that true;
    a timestamp, or a set iterated where a list is meant, would leave two
    shards of the same request differing by something that is not a run.
    """
    written = []
    for name in ("first.json", "second.json"):
        path = tmp_path / name
        code = driver.main(
            [
                "--dry-run",
                "--plan-json",
                str(path),
                "--results",
                str(tmp_path / "none"),
                "--splits",
                str(splits_tree(tmp_path, datasets=("imdb", "spambase"))),
                "--datasets",
                "imdb,spambase",
                *ALL_PRIORS,
            ]
        )
        assert code == 0
        written.append(path.read_bytes())

    assert written[0] == written[1]


# --- a request now names two things: a matrix, and a slice of it -------------


def test_edge_a_custom_matrix_and_a_scope_read_splits_for_their_intersection(
    driver, tmp_path, monkeypatch
):
    """Two filters over one matrix, and the read follows both.

    The matrix decides which datasets this request has and the scope decides
    which of them it runs.  A read that followed only the matrix would open the
    splits of a dataset the request never runs -- and a defect in them would
    stop a shard that does not touch them, which is the failure the dataset
    filter was introduced to prevent.
    """
    custom = _protocol_without(tmp_path, "imdb")  # spambase and cifar10
    seen: list[tuple[str, tuple[str, ...] | None]] = []
    for name in ("split_digests", "population_priors"):
        real = getattr(driver, name)

        def _record(root, *, datasets=None, _name=name, _real=real):
            seen.append((_name, None if datasets is None else tuple(datasets)))
            return _real(root, datasets=datasets)

        monkeypatch.setattr(driver, name, _record)

    code = driver.main(
        [
            "--dry-run",
            "--results",
            str(tmp_path / "none"),
            "--splits",
            str(splits_tree(tmp_path)),
            "--protocol",
            custom,
            "--datasets",
            "spambase",
            "--methods",
            "nnpu",
            *ALL_PRIORS,
        ]
    )

    assert code == 0
    assert seen == [("split_digests", ("spambase",)), ("population_priors", ("spambase",))]


def test_edge_an_unknown_name_is_listed_against_the_matrix_the_request_names(
    driver, unit_calls, tmp_path, capsys
):
    """A refusal offers this matrix's names, not the shipped matrix's.

    The two filters are checked in one breath, so the values a message offers
    have to come from the matrix the request named -- listing the shipped
    matrix's paths here would send the operator to a file this run never opens.
    """
    custom = _protocol_without(tmp_path, "cifar10")  # spambase and imdb only

    code = driver.main(
        [
            "--results",
            str(tmp_path / "out"),
            "--splits",
            str(splits_tree(tmp_path)),
            "--protocol",
            custom,
            "--methods",
            "self_pu",
            "--training-paths",
            "cnn_feature_adapter",
            *ALL_PRIORS,
        ]
    )

    assert code == 1
    assert unit_calls == []
    err = capsys.readouterr().err
    assert "unknown training_path(s) ['cnn_feature_adapter']" in err
    assert "native_2d" in err  # what this matrix has
    assert "native_cnn" not in err  # a path of the shipped matrix, not this one
