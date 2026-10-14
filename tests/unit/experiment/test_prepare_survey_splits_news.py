# ruff: noqa: N803, N806, S101
"""20News keeps all original bytes decoded as latin-1 and the official boundary."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit
CATEGORIES = (
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
)
PREFIXES = ("alt", "comp", "misc", "rec", "sci", "soc", "talk")


@pytest.fixture(scope="module")
def prep_script():
    path = Path(__file__).resolve().parents[3] / "scripts/prepare_survey_splits.py"
    spec = importlib.util.spec_from_file_location("prep_news", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(raw_dir, *, reverse=False):
    for pool in ("train", "test"):
        for category in reversed(CATEGORIES) if reverse else CATEGORIES:
            directory = raw_dir / "twenty_newsgroups" / f"20news-bydate-{pool}" / category
            directory.mkdir(parents=True, exist_ok=True)
            for filename in ("2", "10") if reverse else ("10", "2"):
                (directory / filename).write_bytes(
                    f"From: {pool}/{category}/{filename}\r\n\r\n> quoted".encode()
                    + b"\xff\r\n footer \n"
                )


class _Encoder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **_kwargs):
        self.calls += 1
        values = np.zeros((len(texts), 384), dtype=np.float32)
        values[:, 0] = 1
        values[:, 1] = np.arange(len(texts))
        return values


def test_basic_news_preserves_original_text_and_seven_ids(prep_script, tmp_path):
    _write(tmp_path)
    train, labels, test, test_labels = prep_script.load_twenty_newsgroups(tmp_path)
    assert len(train) == len(test) == 40
    assert train[0] == "From: train/alt.atheism/10\r\n\r\n> quotedÿ\r\n footer \n"
    assert test[0].startswith("From: test/")
    assert (
        labels
        == test_labels
        == [PREFIXES.index(c.split(".")[0]) for c in CATEGORIES for _ in range(2)]
    )


@pytest.mark.parametrize("pool", ["train", "test"])
def test_param_unknown_categories_are_refused(prep_script, tmp_path, pool):
    _write(tmp_path)
    (tmp_path / "twenty_newsgroups" / f"20news-bydate-{pool}" / "unknown.group").mkdir()
    with pytest.raises(ValueError, match="categories"):
        prep_script.load_twenty_newsgroups(tmp_path)


def test_edge_missing_or_empty_category_is_refused(prep_script, tmp_path):
    _write(tmp_path)
    directory = tmp_path / "twenty_newsgroups/20news-bydate-test/alt.atheism"
    for name in ("10", "2"):
        (directory / name).unlink()
    with pytest.raises(ValueError, match="empty"):
        prep_script.load_twenty_newsgroups(tmp_path)
    directory.rmdir()
    with pytest.raises(ValueError, match="categories"):
        prep_script.load_twenty_newsgroups(tmp_path)


def test_determ_news_order_does_not_depend_on_creation_order(prep_script, tmp_path):
    _write(tmp_path / "a")
    _write(tmp_path / "b", reverse=True)
    assert prep_script.load_twenty_newsgroups(tmp_path / "a") == prep_script.load_twenty_newsgroups(
        tmp_path / "b"
    )


def test_basic_news_official_test_and_cache_survive_two_seeds(prep_script, tmp_path):
    _write(tmp_path)
    texts, labels, texts_test, labels_test = prep_script.load_twenty_newsgroups(tmp_path)
    encoder = _Encoder()
    for seed in (0, 1):
        prep_script.prepare_text(
            texts,
            labels,
            texts_test,
            labels_test,
            dataset="twenty_newsgroups",
            seed=seed,
            run_dir=tmp_path / f"split_{seed}",
            cache_dir=tmp_path / "cache",
            encoder=encoder,
        )
    assert encoder.calls == 1
    with (
        np.load(tmp_path / "split_0/test.npz") as first,
        np.load(tmp_path / "split_1/test.npz") as second,
    ):
        np.testing.assert_array_equal(first["X"], second["X"])
        np.testing.assert_array_equal(first["y"], [int(label < 4) for label in labels_test])
    manifest = json.loads((tmp_path / "split_0/split_manifest.json").read_text())
    assert manifest["source_encoding"]["text_encoding"] == "latin-1"
    assert manifest["source_encoding"]["remove"] == []
    assert manifest["source_encoding"]["category_to_class"] == {
        c: PREFIXES.index(c.split(".")[0]) for c in CATEGORIES
    }
    assert "text_encoding" not in manifest["preprocessing"]
