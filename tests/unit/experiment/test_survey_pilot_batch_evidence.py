# tests/unit/experiment/test_survey_pilot_batch_evidence.py

# ruff: noqa: F811

"""The driver's half of the batch evidence: write one packet, cite it, never twice.

The unit script refuses a packet that does not describe the batch, so what these
tests have to hold is that the driver's side of that contract is honest: a
packet is written before the unit script is invoked, the path handed over is a
packet that describes exactly that invocation, a dry run leaves nothing behind,
and a second pass does not manufacture a second packet for the same batch.
"""

from pathlib import Path

import pytest
from _survey_pilot_helpers import (  # noqa: F401 - imported fixtures are used by name
    driver,
    splits_tree,
    unit_calls,
)

from pu_toolbox.experiment import batch_evidence as be
from pu_toolbox.experiment.survey_protocol import PROTOCOL_PATHS, digest, load_protocol
from pu_toolbox.utils.serialization import file_hash

pytestmark = pytest.mark.unit

DATASET = "spambase"
METHOD = "upu"
TRAINING_PATH = "native_2d"
LOCK = Path(__file__).resolve().parents[3] / "uv.lock"

CODE = {"code_commit": "b" * 40, "code_commit_dirty": False, "code_commit_state": "recorded"}
ENVIRONMENT = {"python_version": "3.10.8", "torch_version": "2.13.0+cu130"}


@pytest.fixture
def matrix(driver):
    """The shipped matrix narrowed to one unit, so a batch is cheap to plan."""
    scope = driver.PilotScope(
        datasets=(DATASET,), methods=(METHOD,), training_paths=(TRAINING_PATH,)
    )
    return driver.select_execution_units(load_protocol(PROTOCOL_PATHS["survey-v3.0"]), scope)


@pytest.fixture
def stub(driver, monkeypatch, matrix):
    """Serve the narrowed matrix for whatever path ``--protocol`` resolves to."""
    monkeypatch.setattr(driver, "load_protocol", lambda path: matrix)
    return matrix


@pytest.fixture
def probes(driver, monkeypatch):
    """Fixed identity blocks: a packet must not depend on the machine running it."""
    monkeypatch.setattr(driver.batch_evidence, "code_identity", lambda: dict(CODE))
    monkeypatch.setattr(driver.batch_evidence, "environment_identity", lambda: dict(ENVIRONMENT))
    return driver.batch_evidence


def run(driver, tmp_path, *extra: str, results: Path | None = None) -> int:
    splits = splits_tree(tmp_path / "splits", prior=0.5, datasets=(DATASET,))
    return driver.main(
        [
            "--protocol",
            "survey-v3.0",
            "--splits",
            str(splits),
            "--results",
            str(results or tmp_path / "results"),
            "--datasets",
            DATASET,
            "--methods",
            METHOD,
            "--training-paths",
            TRAINING_PATH,
            *extra,
        ]
    )


def unit_request(argv: list[str]) -> dict:
    """The batch an invocation of the unit script is about, read from its own argv."""
    oracle = "--oracle" in argv

    def value(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    return {
        "unit": {
            "dataset": value("--dataset"),
            "method": "pn_oracle" if oracle else value("--method"),
            "training_path": value("--training-path"),
            "mechanism": None if oracle else value("--labeling-mechanism"),
        },
        "seeds": tuple(int(seed) for seed in value("--seeds").split(",")),
        "c_tokens": () if oracle else tuple(value("--c").split(",")),
    }


def test_basic_every_batch_hands_the_unit_script_a_packet_that_describes_it(
    stub, probes, driver, tmp_path, unit_calls
):
    code = run(driver, tmp_path)

    # The stand-in writes no manifests, so the runs stay pending and the driver
    # reports failure; what matters here is what it did before each invocation.
    assert code == 1
    assert unit_calls
    directory = tmp_path / "results" / be.DIRECTORY
    assert sorted(directory.glob("*.json")) == sorted(
        Path(argv[argv.index("--batch-evidence") + 1]) for argv in unit_calls
    )
    for argv in unit_calls:
        payload = be.load(Path(argv[argv.index("--batch-evidence") + 1]))
        request = unit_request(argv)
        be.require_matches(payload, protocol_sha256=digest(stub), **request)
        assert payload["protocol_version"] == "survey-v3.0"
        # Written out rather than recomputed: scar runs 3 c tokens here and each
        # SAR variant 2, over the protocol's five seeds.
        assert (
            payload["run_count"]
            == {"scar": 15, "sar_lbe_a": 10, "sar_lbe_b": 10}[payload["unit"]["mechanism"]]
        )


def test_basic_the_packet_records_the_lock_this_checkout_ships(
    stub, probes, driver, tmp_path, unit_calls
):
    run(driver, tmp_path)

    payload = be.load(Path(unit_calls[0][unit_calls[0].index("--batch-evidence") + 1]))
    assert payload["dependency_lock"] == {
        "path": "uv.lock",
        "sha256": file_hash(LOCK),
        "state": "recorded",
    }
    assert payload["code_commit"] == CODE["code_commit"]
    assert payload["environment"] == ENVIRONMENT


def test_param_a_dry_run_writes_no_packet_and_starts_no_batch(
    stub, probes, driver, tmp_path, unit_calls
):
    assert run(driver, tmp_path, "--dry-run") == 0

    assert unit_calls == []
    assert not (tmp_path / "results" / be.DIRECTORY).exists()


def test_edge_a_second_pass_leaves_the_packet_alone(stub, probes, driver, tmp_path, unit_calls):
    run(driver, tmp_path)
    directory = tmp_path / "results" / be.DIRECTORY
    before = {path.name: path.read_bytes() for path in directory.glob("*.json")}
    unit_calls.clear()

    run(driver, tmp_path)

    after = {path.name: path.read_bytes() for path in directory.glob("*.json")}
    assert after == before
    assert len(after) == 3


def test_determ_each_mechanism_gets_its_own_packet(stub, probes, driver, tmp_path, unit_calls):
    run(driver, tmp_path)

    payloads = [be.load(Path(argv[argv.index("--batch-evidence") + 1])) for argv in unit_calls]
    identities = {payload["batch_id"] for payload in payloads}
    mechanisms = {payload["unit"]["mechanism"] for payload in payloads}

    assert len(payloads) == 3
    assert len(identities) == 3
    assert mechanisms == {"scar", "sar_lbe_a", "sar_lbe_b"}
    files = {Path(argv[argv.index("--batch-evidence") + 1]).name for argv in unit_calls}
    assert len(files) == 3
