"""Synthetic tiny draft matrices; never canonical values or formal budgets."""

import copy

import numpy as np
from _profile_identity_helpers import draft

from pu_toolbox.experiment.survey_episodic_budget import SCHEMA, schedule
from pu_toolbox.experiment.survey_staged_budget import FIELD


def matrix(algorithm, **overrides):
    params = {
        "robust_pu": dict(
            pretrain_epochs=1,
            episodes=2,
            inner_epochs=2,
            batch_size=8,
            hidden_dim=4,
            pretrain_lr=0.001,
            learning_rate=0.001,
        ),
        "split_pu": dict(
            teacher_epochs=1,
            split_epochs=4,
            student_epochs=2,
            rounds=2,
            batch_size=8,
            hidden_dim=4,
            learning_rate=0.001,
            agreement_threshold=0.5,
        ),
        "holistic_pu": dict(
            warmup_epochs=4,
            max_epochs=2,
            batch_size=8,
            hidden_dim=4,
            learning_rate=0.001,
            warmup_selection="fixed",
            pseudo_pn_initialization="continue",
            lzo_alpha=0.5,
            lzo_validation_size=None,
        ),
    }[algorithm] | overrides
    protocol = draft("nnpu", f"{algorithm}_recipe")
    profile = protocol["method_profiles"][f"{algorithm}_recipe"]
    profile.update(algorithm=algorithm, budget="native_stages", params=copy.deepcopy(params))
    schedule_ = schedule(algorithm, params)
    contract = dict(
        schema_version=SCHEMA,
        algorithm_id=algorithm,
        parameter_values=params,
        checkpoint_epoch_count=sum(count for _, count in schedule_),
        checkpoint_bytes_per_component=2**20,
        eligible_stages=[stage for stage, count in schedule_ if count],
    )
    protocol["budgets"] = {"native_stages": {FIELD: contract}}
    row = protocol["execution_units"][0]
    row.update(
        budget="native_stages",
        model_family=f"{algorithm}_native",
        backbone=f"{algorithm}_native_mlp",
    )
    protocol["backbone_specs"][row["backbone"]] = {
        "algorithm_id": algorithm,
        "parameter_values": {"hidden_dim": params["hidden_dim"]},
    }
    return protocol


def training_data():
    return (
        np.random.RandomState(0).normal(size=(18, 3)).astype("float32"),
        np.r_[np.ones(3, int), np.zeros(15, int)],
    )
