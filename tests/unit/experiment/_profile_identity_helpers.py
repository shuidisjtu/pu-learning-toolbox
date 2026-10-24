"""Private short, noncanonical draft matrices; never production budgets."""

import copy
import json

from pu_toolbox.experiment.survey_profile_identity import FIELD, PROTOCOL_VERSION, SCHEMA
from pu_toolbox.experiment.survey_protocol import load_protocol


def draft(algorithm="upu", profile_id=None):
    protocol = load_protocol()
    profile_id = profile_id or f"{algorithm}_recipe"
    profile = copy.deepcopy(protocol["method_profiles"][algorithm])
    profile["algorithm"] = algorithm
    rows = [
        copy.deepcopy(row)
        for row in protocol["execution_units"]
        if row["method"] == algorithm and row["dataset"] == "spambase"
    ]
    for row in rows:
        row["method"] = profile_id
    protocol.update(
        protocol_version=PROTOCOL_VERSION,
        method_profiles={profile_id: profile},
        execution_units=rows,
        seeds=[0],
        c_tokens={"scar": ["0.5"]},
        candidate_pool=[{}],
    )
    protocol[FIELD] = SCHEMA
    if algorithm == "nnpu":
        profile["params"]["max_epochs"] = 2
        protocol["budgets"][profile["budget"]]["epochs"] = 2
    return protocol


def write_matrix(directory, protocol):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "matrix.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return path
