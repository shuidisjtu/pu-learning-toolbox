# ruff: noqa: N803, N806
"""Split-manifest provenance: what the source states, and what this machine fetched.

P1.2 asks a manifest to carry the data's source, version, label mapping and
licence.  The tests here pin the two halves apart: the publishing facts are
reviewed and live in the catalog, the download digest is machine-local and
travels in the artifact.

The licence half was wrong once, in a way worth pinning.  Every entry was
recorded as "the source states no licence" -- asserted from a partial read of
the distribution pages -- while UCI's page states CC BY 4.0 and its attribution
requirement outright.  An empty ``license_name`` therefore has to mean "the
page names none", not "nobody looked", which is why the check date is required
and the licence assertions below name the actual licence.
"""

import numpy as np
import pytest

from pu_toolbox.experiment import prepare_survey_dataset, survey_dataset_catalog
from pu_toolbox.experiment.datasets import DatasetProvenance

pytestmark = pytest.mark.unit

_SPAMBASE_DOWNLOAD = {
    "sha256": "813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636",
    "bytes": 125537,
    "downloaded_at": "2026-09-08T10:01:00+08:00",
    "source_url": "https://archive.ics.uci.edu/static/public/94/spambase.zip",
}


def _numeric(n_samples=400, n_features=3, n_classes=2):
    rng = np.random.default_rng(12)
    return rng.normal(size=(n_samples, n_features)), np.arange(n_samples) % n_classes


def _spambase_manifest(*, seed=3, **overrides):
    X, y = _numeric()
    _, manifest = prepare_survey_dataset(X, y, dataset="spambase", seed=seed, **overrides)
    return manifest


# --- what the manifest carries ----------------------------------------------


def test_basic_manifest_records_the_reviewed_facts_and_the_fetched_digest():
    provenance = _spambase_manifest(download=_SPAMBASE_DOWNLOAD)["provenance"]

    assert provenance["status"] == "recorded"
    assert provenance["source_url"] == "https://archive.ics.uci.edu/dataset/94/spambase"
    assert provenance["version"] and provenance["citation"]
    assert provenance["label_semantics"]
    assert provenance["download"]["sha256"] == _SPAMBASE_DOWNLOAD["sha256"]
    # The label mapping is the manifest's own positive_classes/negative_classes,
    # one level up: copying it in here would give the two copies room to drift.
    assert "positive_classes" not in provenance
    assert "negative_classes" not in provenance


def test_basic_absent_download_record_is_stated_rather_than_omitted():
    assert _spambase_manifest()["provenance"]["download"] == {"status": "not_recorded"}


def test_basic_a_dataset_without_reviewed_facts_still_reports_its_download():
    """The digest is verified locally, so it survives not having been reviewed."""
    X_source, y_source = _numeric(n_classes=10)
    X_test, y_test = _numeric(n_samples=100, n_classes=10)
    _, manifest = prepare_survey_dataset(
        X_source,
        y_source,
        dataset="mnist",
        seed=0,
        X_test=X_test,
        y_test=y_test,
        download={"sha256": "a" * 64, "bytes": 7, "downloaded_at": "2026-09-08T10:00:00+08:00"},
    )
    assert manifest["provenance"]["status"] == "not_recorded"
    assert manifest["provenance"]["reason"]
    assert manifest["provenance"]["download"]["sha256"] == "a" * 64


def test_edge_the_download_block_names_the_url_the_bytes_came_from():
    """A landing page beside a mirror's digest invites recomputing from the wrong host."""
    record = dict(_SPAMBASE_DOWNLOAD, mirror_url="https://mirror.example/spambase.zip")
    download = _spambase_manifest(download=record)["provenance"]["download"]
    assert download["retrieved_from"] == "https://mirror.example/spambase.zip"


# --- the licence verdict -----------------------------------------------------


def test_edge_a_source_that_states_a_licence_has_it_recorded():
    """UCI states CC BY 4.0; recording "none stated" for it was the P1.2 defect."""
    license_block = _spambase_manifest(download=_SPAMBASE_DOWNLOAD)["provenance"]["license"]
    assert license_block["status"] == "stated"
    assert license_block["name"] == "CC BY 4.0"
    assert license_block["url"] == "https://creativecommons.org/licenses/by/4.0/legalcode"


def test_edge_a_source_that_states_none_records_a_date_and_no_name():
    """The date is what separates "the page names none" from "nobody looked"."""
    catalog = survey_dataset_catalog()
    for name in ("cifar10", "imdb"):
        provenance = catalog[name].provenance
        assert provenance.license_name == "", name
        assert provenance.license_url == "", name
        assert provenance.license_checked_on, name
        assert provenance.license_notes, name


# --- parameter errors --------------------------------------------------------


def test_param_a_download_record_that_cannot_account_for_the_fetch_is_refused():
    X, y = _numeric()
    with pytest.raises(ValueError, match="download record is missing"):
        prepare_survey_dataset(X, y, dataset="spambase", seed=3, download={"sha256": "a" * 64})


def test_param_a_licence_needs_its_url_and_its_check_date():
    fields = {
        "source_url": "u",
        "version": "v",
        "citation": "c",
        "label_semantics": "s",
    }
    with pytest.raises(ValueError, match="needs the URL"):
        DatasetProvenance(**fields, license_checked_on="2026-09-19", license_name="MIT")
    with pytest.raises(ValueError, match="when its licence was checked"):
        DatasetProvenance(**fields, license_checked_on="", license_name="MIT", license_url="u")


# --- determinism -------------------------------------------------------------


def test_determ_provenance_is_the_same_facts_whichever_seed_was_drawn():
    """The artifact set relies on this: all five seeds must carry one provenance block."""
    first = _spambase_manifest(seed=0, download=_SPAMBASE_DOWNLOAD)
    second = _spambase_manifest(seed=4, download=_SPAMBASE_DOWNLOAD)
    assert first["seed"] != second["seed"]
    assert first["provenance"] == second["provenance"]


# --- the catalog itself ------------------------------------------------------


def test_basic_every_pilot_dataset_carries_reviewed_facts():
    """The three the split pipeline builds must have reviewed facts in the catalog."""
    catalog = survey_dataset_catalog()
    for name in ("spambase", "cifar10", "imdb"):
        provenance = catalog[name].provenance
        assert provenance is not None, name
        assert provenance.source_url.startswith("https://")
        assert provenance.version and provenance.citation and provenance.label_semantics
        assert provenance.license_notes
        assert bool(provenance.license_name) == bool(provenance.license_url)
