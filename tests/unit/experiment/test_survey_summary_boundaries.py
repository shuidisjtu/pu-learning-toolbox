"""What a report row refuses to average together.

A mean over two incomparable things reads exactly like a mean over two
comparable ones, so every boundary here is asserted as "these two manifests do
not share a row key" rather than left to the aggregation gate upstream.  The gate
decides which runs may be *compared*; the row key decides which numbers get
*averaged*, and those are different questions asked at different times.

``training_path`` and ``dataset`` are deliberately absent from the key: the
comparability group already pins both, and the aggregation entry point refuses a
group whose members disagree about either.  Listing them again would create a
second source for the same fact -- the shape that lets one reader and another
disagree about whether two runs are in the same row.
"""

import pytest

from pu_toolbox.experiment.survey_summary import row_key

pytestmark = pytest.mark.unit


def _manifest(
    *,
    method="nnpu",
    dataset="spambase",
    training_path="native_2d",
    budget_family="minibatch",
    group=None,
    run_view="ts-compatible",
    mechanism="scar",
    c=0.1,
    c_independent=False,
):
    """A manifest whose group string and direct fields agree, as a real one does."""
    payload = {
        "execution_unit": {
            "method": method,
            "dataset": dataset,
            "comparability_group": group or f"{dataset}/{training_path}/{budget_family}",
        },
        "run_view": run_view,
        "training_path": training_path,
        "c_requested_token": None if c_independent else f"{c:g}",
        "generation": {
            "train": {"mechanism": "pn_oracle" if c_independent else mechanism, "c_requested": c}
        },
    }
    if c_independent:
        payload["c_independent"] = True
    return payload


def _key(manifest, protocol="OA"):
    return row_key(manifest, selection_protocol=protocol)


def test_basic_two_views_of_one_method_never_share_a_row_key():
    os_row = _key(_manifest(run_view="os-compatible"))
    ts_row = _key(_manifest(run_view="ts-compatible"))

    assert os_row != ts_row
    # And the difference is the view, not something incidental.
    assert os_row[1] == "os-compatible"
    assert ts_row[1] == "ts-compatible"


def test_basic_scar_and_sar_never_share_a_row_key_even_at_a_shared_c():
    # SCAR covers {0.1, 0.3, 0.5} and SAR {0.05, 0.5}, so c = 0.5 is in both.
    scar_row = _key(_manifest(mechanism="scar", c=0.5))
    sar_row = _key(_manifest(mechanism="sar_lbe_a", c=0.5))

    assert scar_row != sar_row
    assert scar_row[2] == "scar"
    assert sar_row[2] == "sar_lbe_a"


def test_basic_native_cnn_and_feature_adapter_never_share_a_row_key():
    native = _key(_manifest(training_path="native_cnn", budget_family="fullbatch"))
    adapter = _key(_manifest(training_path="cnn_feature_adapter", budget_family="fullbatch"))

    assert native != adapter


def test_basic_the_oracle_gets_a_row_of_its_own():
    oracle = _key(_manifest(method="pn_oracle", mechanism="pn_oracle", c=0.1, c_independent=True))
    ordinary = _key(_manifest(method="nnpu", mechanism="scar", c=0.1))

    assert oracle != ordinary
    assert oracle[-1] == "c_independent"
    # The oracle is OA only; a PA row for it would be a result it never produced.
    assert _key(_manifest(c_independent=True), protocol="PA") != oracle


def test_basic_pa_and_oa_are_separate_rows_of_the_same_run():
    manifest = _manifest()

    pa_row, oa_row = _key(manifest, "PA"), _key(manifest, "OA")

    assert pa_row != oa_row
    # They differ in the protocol slot alone -- same group, view, mechanism,
    # method and c -- so the two rows are one run reported twice, not two runs.
    assert pa_row[:4] == oa_row[:4]
    assert pa_row[5] == oa_row[5]
    assert {pa_row[4], oa_row[4]} == {"PA", "OA"}


def test_param_a_different_dataset_changes_the_row_key():
    spambase = _key(_manifest(dataset="spambase"))
    imdb = _key(_manifest(dataset="imdb"))

    assert spambase != imdb


def test_edge_two_budget_families_of_one_dataset_stay_apart():
    # dist_pu and self_pu live in different budget families, so they are not one
    # leaderboard even on the same dataset and path.
    fullbatch = _key(_manifest(method="dist_pu", budget_family="fullbatch"))
    sampled = _key(_manifest(method="self_pu", budget_family="two_student_sampled"))

    assert fullbatch != sampled


def test_edge_the_view_used_is_read_from_the_manifest_not_the_method():
    # A method native to TS but not wired for calibration falls back to
    # os-compatible; the row must follow the view that ran, not the method's
    # declared assumption.
    fell_back = _key(_manifest(method="pusb_kernel", run_view="os-compatible"))
    calibrated = _key(_manifest(method="pusb_kernel", run_view="ts-compatible"))

    assert fell_back[1] == "os-compatible"
    assert fell_back != calibrated


def test_determ_the_row_key_is_a_pure_function_of_the_manifest():
    manifest = _manifest()

    assert _key(manifest) == _key(manifest)
    # Rebuilding the payload in a different key order changes nothing: the key is
    # read by name, not by position.
    shuffled = dict(reversed(list(manifest.items())))
    assert _key(shuffled) == _key(manifest)
