# tests/unit/estimators/test_self_pu_ts_view.py

# ruff: noqa: E501, E402, N803, N806, S101, S311

"""The ``os_or_ts`` training view on Self-PU, and the frozen OS baseline.

Protocol §2.3 replaces a native-TS method's unlabeled-loss input with
``D_U ∪ D_P`` per training mini-batch.  Self-PU cannot take that literally: its
negative term is drawn from the *untrusted* share of the U batch, so the
calibrated role set is "the U rows already carrying the negative role, plus the
positive batch", and the two sides are blended by row mass rather than by a
fixed half.  These tests pin that blend, the fail-loud view flag, and the
pre-refactor OS trajectory.

Two configuration shapes are frozen, because they are different code paths:
``A`` is the Pilot's (PU validation only, so meta reweighting is inactive and
the weights are uniform), ``B`` is the library's (clean validation, so the meta
weights are active and non-uniform).  The golden block is a *characterization*
freeze captured from the code before the refactor: it is green on arrival by
design, and it is what makes "the OS path did not drift" checkable afterwards.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from pu_toolbox.estimators.deep import self_pu as self_pu_module  # noqa: E402
from pu_toolbox.estimators.deep.self_pu import SelfPUClassifier  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.filterwarnings("ignore:validation_data was not supplied"),
]

_RTOL = 1e-6
_ATOL = 1e-9

# Sentinel: "call fit without os_or_ts at all", which is not the same request as
# passing any particular value -- including None, which is an invalid view.
_DEFAULT = object()

_CONF = dict(
    class_prior=1 / 3,
    hidden_dim=4,
    warmup_epochs=0,
    self_paced_start=0,
    self_paced_end=2,
    distill_start=2,
    max_epochs=4,
    max_trust_ratio=0.2,
    pace_1=0.1,
    pace_2=0.2,
    batch_size=16,
    random_state=5,
    device="cpu",
)

_PROBE = np.array(
    [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0], [0.0, 0.5, -0.5, 0.25]],
    dtype=np.float32,
)

_NETS = ("student_1_", "student_2_", "teacher_1_", "teacher_2_")

_GOLDEN = {
    "A": {
        "epochs": [1, 2, 3, 4],
        "probe": [0.7789386510848999, 0.5368010997772217, 0.5770938396453857],
        "state": {
            "student_1_": {
                "1.weight": [
                    0.33274635672569275,
                    -0.37143000960350037,
                    0.40997031331062317,
                    0.3223956525325775,
                    0.417636513710022,
                    -0.385751336812973,
                    -0.33804717659950256,
                    0.23547932505607605,
                    -0.4690532982349396,
                    0.49194610118865967,
                    0.1039978638291359,
                    0.06229022890329361,
                    -0.42508968710899353,
                    0.16182084381580353,
                    0.21750229597091675,
                    0.08182346075773239,
                ],
                "1.bias": [
                    0.48335739970207214,
                    0.15265795588493347,
                    -0.44579511880874634,
                    0.41761672496795654,
                ],
                "3.weight": [
                    0.17226096987724304,
                    -0.24076372385025024,
                    -0.4604162275791168,
                    0.2825588583946228,
                ],
                "3.bias": [0.47266003489494324],
            },
            "student_2_": {
                "1.weight": [
                    0.33273831009864807,
                    -0.37144142389297485,
                    0.40996360778808594,
                    0.32239243388175964,
                    0.41764119267463684,
                    -0.38578924536705017,
                    -0.33806732296943665,
                    0.23546205461025238,
                    -0.4690122902393341,
                    0.4920124411582947,
                    0.10401421785354614,
                    0.062338896095752716,
                    -0.42509767413139343,
                    0.1618127077817917,
                    0.21749575436115265,
                    0.0818154588341713,
                ],
                "1.bias": [
                    0.4833380877971649,
                    0.15265929698944092,
                    -0.44577789306640625,
                    0.4176184833049774,
                ],
                "3.weight": [
                    0.1722530871629715,
                    -0.24087265133857727,
                    -0.4605260491371155,
                    0.2825430929660797,
                ],
                "3.bias": [0.4726601541042328],
            },
            "teacher_1_": {
                "1.weight": [
                    0.3303278982639313,
                    -0.37381401658058167,
                    0.40754589438438416,
                    0.3200024366378784,
                    0.4200281798839569,
                    -0.38342827558517456,
                    -0.33568301796913147,
                    0.2378447949886322,
                    -0.4668068289756775,
                    0.49411505460739136,
                    0.10635421425104141,
                    0.06450075656175613,
                    -0.42751172184944153,
                    0.1594146490097046,
                    0.2150818407535553,
                    0.07940363138914108,
                ],
                "1.bias": [
                    0.4810014069080353,
                    0.15023481845855713,
                    -0.4434404671192169,
                    0.4200356602668762,
                ],
                "3.weight": [
                    0.16984085738658905,
                    -0.23859615623950958,
                    -0.45932629704475403,
                    0.2849601209163666,
                ],
                "3.bias": [0.47508513927459717],
            },
            "teacher_2_": {
                "1.weight": [
                    0.3303276598453522,
                    -0.37381434440612793,
                    0.40754571557044983,
                    0.3200022876262665,
                    0.4200282692909241,
                    -0.38342922925949097,
                    -0.33568358421325684,
                    0.23784436285495758,
                    -0.4668058156967163,
                    0.4941166043281555,
                    0.10635462403297424,
                    0.06450197100639343,
                    -0.42751190066337585,
                    0.15941445529460907,
                    0.21508169174194336,
                    0.07940342277288437,
                ],
                "1.bias": [
                    0.48100101947784424,
                    0.15023483335971832,
                    -0.4434400498867035,
                    0.4200356602668762,
                ],
                "3.weight": [
                    0.16984066367149353,
                    -0.23859988152980804,
                    -0.45932886004447937,
                    0.2849597930908203,
                ],
                "3.bias": [0.47508513927459717],
            },
        },
        "best": (2, 4),
        "basis": "pu_validation_nnpu_risk",
        "mode": "ablation",
        "metrics": [0.5196704268455505, 0.5196703672409058],
        "trusted": [
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 2, 3, 2, 1, 1, 2, 0],
            [2, 1, 3, 2, 1, 1, 2, 0],
            [2, 2, 6, 6, 3, 3, 4, 0],
            [3, 1, 3, 2, 1, 1, 0, 0],
            [3, 2, 6, 6, 3, 3, 0, 0],
            [4, 1, 3, 2, 1, 1, 0, 0],
            [4, 2, 6, 6, 3, 3, 0, 0],
        ],
        "trusted_ids": {
            1: [17, 8],
            2: [17, 22, 27, 11, 7, 8],
        },
        "trusted_soft": {
            1: [0.5757626295089722, 0.7059811353683472],
            2: [
                0.5757356882095337,
                0.5802117586135864,
                0.5859981179237366,
                0.6922352313995361,
                0.7041386365890503,
                0.7059778571128845,
            ],
        },
        "reweight": [
            [1, 1, False, 0.0, 0.0, 1.0, 0.0, False, False],
            [1, 2, False, 0.0, 0.0, 1.0, 0.0, False, False],
            [2, 1, False, 0.0, 0.0, 1.0, 0.0, False, False],
            [2, 2, False, 0.0, 0.0, 1.0, 0.0, False, False],
            [3, 1, False, 0.0, 0.0, 1.0, 0.0, False, False],
            [3, 2, False, 0.0, 0.0, 1.0, 0.0, False, False],
            [4, 1, False, 0.0, 0.0, 1.0, 0.0, False, False],
            [4, 2, False, 0.0, 0.0, 1.0, 0.0, False, False],
        ],
        "train": [
            [1, 1, 0.532006561756134, 0.0, 0.0, 0.0, 0.0, 0.532006561756134, 0.4261758625507355],
            [
                1,
                2,
                0.5286124348640442,
                0.6069241762161255,
                0.0,
                0.0,
                0.0,
                1.1355366706848145,
                0.42278173565864563,
            ],
            [
                2,
                1,
                0.524375855922699,
                0.6064029335975647,
                0.0,
                2.368475909392639e-16,
                6.6642741103351e-07,
                1.130779504776001,
                0.4187050759792328,
            ],
            [
                2,
                2,
                0.5212307572364807,
                0.627932071685791,
                0.0,
                0.0,
                6.6642741103351e-07,
                1.1491634845733643,
                0.4155599772930145,
            ],
            [
                3,
                1,
                0.5185904502868652,
                0.6059695482254028,
                0.0,
                1.892779266654543e-10,
                2.0771926756424364e-06,
                1.1245620250701904,
                0.41305363178253174,
            ],
            [
                3,
                2,
                0.5179046392440796,
                0.6304939389228821,
                0.0,
                1.7459619860993314e-10,
                2.093144530590507e-06,
                1.1484007835388184,
                0.41236406564712524,
            ],
            [
                4,
                1,
                0.5206699967384338,
                0.6057109832763672,
                0.0,
                1.2679776673074628e-10,
                3.4847780625568703e-06,
                1.1263843774795532,
                0.41521185636520386,
            ],
            [
                4,
                2,
                0.5215578079223633,
                0.6377673149108887,
                0.0,
                4.9755755071601016e-11,
                3.5104842481814558e-06,
                1.159328579902649,
                0.4160943627357483,
            ],
        ],
        "distill": [
            [1, 1, False, 0.0, 0.0, 0.0],
            [1, 2, False, 0.0, 0.0, 0.0],
            [2, 1, True, 1.0, 2.368475909392639e-16, 6.6642741103351e-07],
            [2, 2, True, 1.0, 0.0, 6.6642741103351e-07],
            [3, 1, True, 1.0, 1.892779266654543e-10, 2.0771926756424364e-06],
            [3, 2, True, 1.0, 1.7459619860993314e-10, 2.093144530590507e-06],
            [4, 1, True, 1.0, 1.2679776673074628e-10, 3.4847780625568703e-06],
            [4, 2, True, 1.0, 4.9755755071601016e-11, 3.5104842481814558e-06],
        ],
        "valid": {
            "epoch": [1, 2, 3, 4],
            "train_risk": [
                0.5303094983100891,
                0.5228033065795898,
                0.5182475447654724,
                0.5211139023303986,
            ],
            "val_risk": [
                0.5197252035140991,
                0.5197098255157471,
                0.5196905136108398,
                0.5196703672409058,
            ],
            "val_teacher": [1, 1, 2, 2],
            "teacher_1_val_risk": [
                0.5197252035140991,
                0.5197098255157471,
                0.519690752029419,
                0.5196704268455505,
            ],
            "teacher_2_val_risk": [
                0.5197252035140991,
                0.5197098255157471,
                0.5196905136108398,
                0.5196703672409058,
            ],
        },
    },
    "B": {
        "epochs": [1, 2, 3, 4],
        "probe": [0.7789416313171387, 0.536799430847168, 0.5770946145057678],
        "state": {
            "student_1_": {
                "1.weight": [
                    0.3327527642250061,
                    -0.3713870644569397,
                    0.40996500849723816,
                    0.32242730259895325,
                    0.41760164499282837,
                    -0.3858545422554016,
                    -0.33810436725616455,
                    0.2354254275560379,
                    -0.4692350924015045,
                    0.49168503284454346,
                    0.10392880439758301,
                    0.06207152083516121,
                    -0.42508843541145325,
                    0.16184163093566895,
                    0.21750403940677643,
                    0.08182815462350845,
                ],
                "1.bias": [
                    0.48341792821884155,
                    0.1526564061641693,
                    -0.44586583971977234,
                    0.4176158308982849,
                ],
                "3.weight": [
                    0.17226460576057434,
                    -0.2409684807062149,
                    -0.4567889869213104,
                    0.2825719714164734,
                ],
                "3.bias": [0.47266125679016113],
            },
            "student_2_": {
                "1.weight": [
                    0.3327539265155792,
                    -0.37138622999191284,
                    0.40996620059013367,
                    0.32242584228515625,
                    0.4176029562950134,
                    -0.38585707545280457,
                    -0.33810344338417053,
                    0.23542575538158417,
                    -0.4692350924015045,
                    0.49168503284454346,
                    0.10392880439758301,
                    0.06207151710987091,
                    -0.42508718371391296,
                    0.16184072196483612,
                    0.21750091016292572,
                    0.08182783424854279,
                ],
                "1.bias": [
                    0.4834090769290924,
                    0.15265920758247375,
                    -0.44586583971977234,
                    0.417617529630661,
                ],
                "3.weight": [
                    0.17226403951644897,
                    -0.24100473523139954,
                    -0.4567889869213104,
                    0.2825356125831604,
                ],
                "3.bias": [0.4726620018482208],
            },
            "teacher_1_": {
                "1.weight": [
                    0.3303280770778656,
                    -0.37381288409233093,
                    0.4075457453727722,
                    0.3200032114982605,
                    0.4200272262096405,
                    -0.38343092799186707,
                    -0.33568456768989563,
                    0.23784340918064117,
                    -0.46681150794029236,
                    0.4941085875034332,
                    0.10635240375995636,
                    0.06449508666992188,
                    -0.42751166224479675,
                    0.15941518545150757,
                    0.21508190035820007,
                    0.07940378040075302,
                ],
                "1.bias": [
                    0.4810029864311218,
                    0.15023474395275116,
                    -0.44344234466552734,
                    0.4200356602668762,
                ],
                "3.weight": [
                    0.169841006398201,
                    -0.23860155045986176,
                    -0.45920872688293457,
                    0.28496041893959045,
                ],
                "3.bias": [0.47508516907691956],
            },
            "teacher_2_": {
                "1.weight": [
                    0.330328106880188,
                    -0.37381288409233093,
                    0.407545804977417,
                    0.3200031518936157,
                    0.4200272560119629,
                    -0.38343101739883423,
                    -0.335684597492218,
                    0.23784340918064117,
                    -0.46681150794029236,
                    0.4941085875034332,
                    0.10635240375995636,
                    0.06449508666992188,
                    -0.42751166224479675,
                    0.15941517055034637,
                    0.2150818407535553,
                    0.07940376549959183,
                ],
                "1.bias": [
                    0.4810027778148651,
                    0.15023483335971832,
                    -0.44344234466552734,
                    0.4200356602668762,
                ],
                "3.weight": [
                    0.1698409765958786,
                    -0.23860272765159607,
                    -0.45920872688293457,
                    0.284959614276886,
                ],
                "3.bias": [0.47508516907691956],
            },
        },
        "best": (1, None),
        "basis": "clean_validation_accuracy",
        "mode": "clean_validation_meta",
        "metrics": [0.5, 0.5],
        "trusted": [
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 2, 3, 2, 1, 1, 2, 0],
            [2, 1, 3, 2, 1, 1, 2, 0],
            [2, 2, 6, 6, 3, 3, 4, 0],
            [3, 1, 3, 2, 1, 1, 0, 0],
            [3, 2, 6, 6, 3, 3, 0, 0],
            [4, 1, 3, 2, 1, 1, 0, 0],
            [4, 2, 6, 6, 3, 3, 0, 0],
        ],
        "trusted_ids": {
            1: [17, 8],
            2: [17, 22, 27, 11, 7, 8],
        },
        "trusted_soft": {
            1: [0.575718343257904, 0.7060053944587708],
            2: [
                0.5757102370262146,
                0.5801795721054077,
                0.5859639048576355,
                0.6922562122344971,
                0.7041505575180054,
                0.7060016393661499,
            ],
        },
        "reweight": [
            [1, 1, True, 0.0625, 1.0, 1.0, 0.3125, True, False],
            [1, 2, True, 0.06666666666666667, 0.9375, 1.0, 0.26666666666666666, True, False],
            [2, 1, True, 0.06666666666666667, 0.9375, 0.9999999999999999, 0.2, True, False],
            [2, 2, True, 0.08333333333333333, 0.75, 1.0, 0.08333333333333333, True, False],
            [3, 1, True, 0.06666666666666667, 0.9375, 1.0, 0.2, True, False],
            [
                3,
                2,
                True,
                0.07692307692307693,
                0.8125,
                0.9999999999999999,
                0.15384615384615385,
                True,
                False,
            ],
            [4, 1, True, 0.06666666666666667, 0.9375, 1.0, 0.13333333333333333, True, False],
            [4, 2, True, 0.09090909090909091, 0.6875, 1.0000000000000002, 0.0, True, False],
        ],
        "train": [
            [
                1,
                1,
                0.5182141661643982,
                0.0,
                0.60811448097229,
                0.0,
                0.0,
                1.126328706741333,
                0.41238346695899963,
            ],
            [
                1,
                2,
                0.5182141661643982,
                0.6069241762161255,
                0.5701073408126831,
                0.0,
                0.0,
                1.695245623588562,
                0.41238346695899963,
            ],
            [
                2,
                1,
                0.5111410617828369,
                0.6064029335975647,
                0.6231244802474976,
                2.368475909392639e-16,
                6.6642741103351e-07,
                1.7406691312789917,
                0.40547478199005127,
            ],
            [
                2,
                2,
                0.5167322158813477,
                0.627932071685791,
                0.4984995722770691,
                0.0,
                6.6642741103351e-07,
                1.6431646347045898,
                0.411065936088562,
            ],
            [
                3,
                1,
                0.5113299489021301,
                0.6059538125991821,
                0.617311954498291,
                1.5524885196849247e-11,
                2.1995833776600193e-06,
                1.73459792137146,
                0.40580523014068604,
            ],
            [
                3,
                2,
                0.5176373720169067,
                0.6304886341094971,
                0.5350037813186646,
                1.4464736700081637e-11,
                2.207214947702596e-06,
                1.6831320524215698,
                0.41211166977882385,
            ],
            [
                4,
                1,
                0.5071893930435181,
                0.6056897640228271,
                0.6041436791419983,
                2.4431774311994836e-11,
                3.559898686944507e-06,
                1.7170264720916748,
                0.401747465133667,
            ],
            [
                4,
                2,
                0.5195210576057434,
                0.637762188911438,
                0.4430387020111084,
                1.8342338065080455e-11,
                3.5731920888792956e-06,
                1.600325584411621,
                0.4140774607658386,
            ],
        ],
        "distill": [
            [1, 1, False, 0.0, 0.0, 0.0],
            [1, 2, False, 0.0, 0.0, 0.0],
            [2, 1, True, 1.0, 2.368475909392639e-16, 6.6642741103351e-07],
            [2, 2, True, 1.0, 0.0, 6.6642741103351e-07],
            [3, 1, True, 1.0, 1.5524885196849247e-11, 2.1995833776600193e-06],
            [3, 2, True, 1.0, 1.4464736700081637e-11, 2.207214947702596e-06],
            [4, 1, True, 1.0, 2.4431774311994836e-11, 3.559898686944507e-06],
            [4, 2, True, 1.0, 1.8342338065080455e-11, 3.5731920888792956e-06],
        ],
        "valid": {
            "epoch": [1, 2, 3, 4],
            "train_risk": [
                0.5182141661643982,
                0.5139366388320923,
                0.5144836604595184,
                0.5133552253246307,
            ],
            "val_risk": [],
            "val_teacher": [],
            "teacher_1_val_risk": [],
            "teacher_2_val_risk": [],
        },
    },
}


def _data():
    rng = np.random.RandomState(17)
    X = np.vstack([rng.normal(1.0, 0.4, (12, 4)), rng.normal(-1.0, 0.4, (24, 4))]).astype(
        np.float32
    )
    y_pu = np.r_[np.ones(6, dtype=int), np.zeros(30, dtype=int)]
    X_val = np.vstack([rng.normal(1.0, 0.4, (6, 4)), rng.normal(-1.0, 0.4, (6, 4))]).astype(
        np.float32
    )
    y_val = np.r_[np.ones(6, dtype=int), np.zeros(6, dtype=int)]
    y_pu_val = np.r_[np.ones(3, dtype=int), np.zeros(9, dtype=int)]
    return X, y_pu, X_val, y_val, y_pu_val


def _fit(config_key, view=_DEFAULT, **overrides):
    """Fit one frozen configuration; ``_DEFAULT`` leaves the flag at its default.

    A distinct sentinel rather than ``None``, because passing ``None`` explicitly
    is one of the invalid values under test.
    """
    X, y_pu, X_val, y_val, y_pu_val = _data()
    if config_key == "A":
        kwargs = {"pu_validation_data": (X_val, y_pu_val)}
    else:
        kwargs = {"validation_data": (X_val, y_val)}
    if view is not _DEFAULT:
        kwargs["os_or_ts"] = view
    epochs = []
    model = SelfPUClassifier(**{**_CONF, **overrides})
    model.fit(X, y_pu, epoch_callback=lambda epoch, _est: epochs.append(epoch), **kwargs)
    return model, epochs


def _observed(model, epochs):
    """Everything the freeze records, in a shape the matcher can walk."""
    return {
        "epochs": epochs,
        "probe": [float(v) for v in model.decision_function(_PROBE)],
        "state": {
            name: {
                key: [float(x) for x in tensor.flatten()]
                for key, tensor in getattr(model, name).state_dict().items()
            }
            for name in _NETS
        },
        "best": (model.best_teacher_index_, model.best_epoch_),
        "basis": model.teacher_selection_basis_,
        "mode": model.calibration_mode_,
        "metrics": [float(v) for v in model.teacher_selection_metrics_],
        "trusted": [
            [
                record["epoch"],
                record["student"],
                record["target_size"],
                record["actual_size"],
                record["positive_count"],
                record["negative_count"],
                record["entered_count"],
                record["exited_count"],
            ]
            for record in model.trusted_history_
        ],
        "trusted_ids": {
            key: value["indices"].tolist() for key, value in model.trusted_indices_.items()
        },
        "trusted_soft": {
            key: [float(x) for x in value["soft_labels"]]
            for key, value in model.trusted_indices_.items()
        },
        "reweight": [
            [
                record["epoch"],
                record["student"],
                record["calibration_active"],
                float(record["ce_active_fraction"]),
                float(record["ce_weight_sum"]),
                float(record["pu_weight_sum"]),
                float(record["zero_weight_fraction"]),
                record["ce_fallback"],
                record["pu_fallback"],
            ]
            for record in model.reweight_history_
        ],
        "train": [
            [
                record["epoch"],
                record["student"],
                float(record["nnpu_loss"]),
                float(record["trusted_ce"]),
                float(record["calibrated_ce"]),
                float(record["student_consistency"]),
                float(record["teacher_consistency"]),
                float(record["total_loss"]),
                float(record["negative_risk"]),
            ]
            for record in model.training_history_
        ],
        "distill": [
            [
                record["epoch"],
                record["student"],
                record["active"],
                float(record["hard_sample_fraction"]),
                float(record["student_mse"]),
                float(record["teacher_mse"]),
            ]
            for record in model.distillation_history_
        ],
        "valid": {
            key: [int(v) if isinstance(v, (int, np.integer)) else float(v) for v in values]
            for key, values in model.history_.items()
        },
    }


def _assert_matches(observed, expected, path="golden"):
    """Discrete structure exactly, measured values within a tensor tolerance."""
    if isinstance(expected, dict):
        assert set(observed) == set(expected), path
        for key in expected:
            _assert_matches(observed[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert len(observed) == len(expected), path
        for index, (item, reference) in enumerate(zip(observed, expected, strict=True)):
            _assert_matches(item, reference, f"{path}[{index}]")
        return
    if isinstance(expected, (bool, int, str)) or expected is None:
        assert observed == expected, path
        return
    assert observed == pytest.approx(expected, rel=_RTOL, abs=_ATOL), path


class TestFrozenOsBaseline:
    """The pre-refactor OS trajectory, on both configuration shapes."""

    @pytest.mark.parametrize(
        ("config_key", "view"),
        [("A", _DEFAULT), ("A", "os"), ("B", _DEFAULT), ("B", "os")],
        ids=["A-default", "A-explicit", "B-default", "B-explicit"],
    )
    def test_determ_os_golden_matches_the_pre_refactor_run(self, config_key, view):
        """Adding a view flag must not move the view it defaults to."""
        model, epochs = _fit(config_key, view)
        _assert_matches(_observed(model, epochs), _GOLDEN[config_key])

    @pytest.mark.parametrize("config_key", ["A", "B"])
    def test_determ_the_default_view_is_the_explicit_os_view(self, config_key):
        """Not "close to": the default and an explicit OS are the same run.

        A tolerance would hide a default that silently drifted to the calibrated
        path, which is exactly the failure this pair exists to catch.
        """
        default, default_epochs = _fit(config_key)
        explicit, explicit_epochs = _fit(config_key, "os")

        assert _observed(default, default_epochs) == _observed(explicit, explicit_epochs)

    @pytest.mark.parametrize("config_key", ["A", "B"])
    def test_basic_the_ts_view_fits_and_moves_the_solution(self, config_key):
        """The flag has to reach the math, not merely be accepted."""
        os_model, os_epochs = _fit(config_key, "os")
        ts_model, ts_epochs = _fit(config_key, "ts")

        assert ts_epochs == os_epochs
        assert ts_model._X_shape_ == os_model._X_shape_
        assert [record["nnpu_loss"] for record in ts_model.training_history_] != [
            record["nnpu_loss"] for record in os_model.training_history_
        ]
        assert np.isfinite(ts_model.decision_function(_PROBE)).all()


class TestBlendedNegativeRole:
    """The calibrated negative term, against literals computed by hand."""

    # Six U rows already carrying the negative role, three positive rows.
    _U_LOSSES = [0.9, 0.7, 0.2, 0.1, 0.4, 0.6]
    _U_WEIGHTS = [0.1, 0.2, 0.3, 0.1, 0.2, 0.1]
    _P_LOSSES = [0.05, 0.15, 0.35]

    def _u_side(self, weights):
        return float(
            torch.tensor(
                [w * loss for w, loss in zip(weights, self._U_LOSSES, strict=True)],
                dtype=torch.float64,
            ).sum()
        )

    def test_basic_the_blend_matches_independently_computed_literals(self):
        """Uniform U weights make the blend the plain union mean; non-uniform do not.

        The first identity is the whole justification for calibrating this term:
        with the U side already a mean, blending by row count *is* the empirical
        mean over the role union.  The second shows the blend is not claiming to
        be that mean once the meta weights re-weight the U side.
        """
        uniform = [1 / 6] * 6
        uniform_u = self._u_side(uniform)  # 0.4833333...
        non_uniform_u = self._u_side(self._U_WEIGHTS)  # 0.44
        positive = torch.tensor(self._P_LOSSES, dtype=torch.float64)

        blended_uniform = self_pu_module._marginal_negative_risk(
            uniform_u, positive, n_unlabeled_role=6, include_positive_in_unlabeled=True
        )
        blended_weighted = self_pu_module._marginal_negative_risk(
            non_uniform_u, positive, n_unlabeled_role=6, include_positive_in_unlabeled=True
        )
        untouched = self_pu_module._marginal_negative_risk(
            non_uniform_u, positive, n_unlabeled_role=6, include_positive_in_unlabeled=False
        )

        # Gate A: the no-meta blend is the empirical mean over U_role ∪ P.
        union_mean = (sum(self._U_LOSSES) + sum(self._P_LOSSES)) / 9
        assert float(blended_uniform) == pytest.approx(union_mean, rel=1e-12)
        # With meta weights it is the stated mixture, and not that mean.
        expected = (6 / 9) * non_uniform_u + (3 / 9) * (sum(self._P_LOSSES) / 3)
        assert float(blended_weighted) == pytest.approx(expected, rel=1e-12)
        assert float(blended_weighted) != pytest.approx(union_mean, rel=1e-12)
        # OS is untouched, not merely equal after arithmetic.
        assert float(untouched) == non_uniform_u

    @pytest.mark.parametrize(
        ("n_unlabeled_role", "n_positive"),
        [(6, 3), (12, 3), (5, 9)],
        ids=["u-heavy", "u-dominant", "p-heavy"],
    )
    def test_param_the_blend_weights_are_row_mass_not_a_fixed_half(
        self, n_unlabeled_role, n_positive
    ):
        """Unequal sides reproduce the union mean; a fixed half would not.

        The expected value is the mean of an explicitly built union array, so it
        does not restate the blend's own formula.
        """
        uniform_u = self._u_side([1 / 6] * 6)
        positives = [0.05] * n_positive
        union = np.r_[np.full(n_unlabeled_role, uniform_u), np.array(positives)]
        fixed_half = (uniform_u + sum(positives) / n_positive) / 2

        blended = self_pu_module._marginal_negative_risk(
            uniform_u,
            torch.tensor(positives, dtype=torch.float64),
            n_unlabeled_role=n_unlabeled_role,
            include_positive_in_unlabeled=True,
        )
        untouched = self_pu_module._marginal_negative_risk(
            uniform_u,
            torch.tensor(positives, dtype=torch.float64),
            n_unlabeled_role=n_unlabeled_role,
            include_positive_in_unlabeled=False,
        )

        assert float(blended) == pytest.approx(union.mean(), rel=1e-12)
        assert float(blended) != pytest.approx(fixed_half, rel=1e-12)
        assert float(untouched) == uniform_u


class TestViewFlag:
    """The flag's shape, its default, and its failure mode."""

    def test_param_the_view_flag_is_keyword_only_and_defaults_to_os(self):
        parameters = inspect.signature(SelfPUClassifier.fit).parameters

        assert parameters["os_or_ts"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["os_or_ts"].default == "os"

    @pytest.mark.parametrize("value", ["TS", "strict", "", None, "os_or_ts"])
    def test_param_an_unknown_view_value_is_refused(self, value):
        """A typo must not silently train the uncalibrated view."""
        with pytest.raises(ValueError, match="os_or_ts must be 'os' or 'ts'"):
            _fit("A", value)

    def test_edge_an_unknown_view_value_does_not_perturb_the_global_seed(self):
        """Validation precedes any RNG use, so a rejected call leaves no trace.

        The reference draw is taken *before* the rejected call and the next draw
        after it is compared against that reference: re-seeding before each
        comparison would make this pass no matter what the call did.
        """
        torch.manual_seed(1234)
        reference = torch.rand(3)

        torch.manual_seed(1234)
        with pytest.raises(ValueError, match="os_or_ts must be 'os' or 'ts'"):
            _fit("A", "strict")

        assert torch.equal(torch.rand(3), reference)


class _FirstFeatureBackbone(nn.Module):
    """A backbone whose score is monotone in the first feature.

    Lets a test place the known positives at the *bottom* of the ranking — the
    one arrangement in which a trusted-set manager that could see them would
    hand them a trusted-negative label.  The scale keeps one real parameter on
    the module, so the optimizer is not handed an empty parameter list.
    """

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, X):
        return self.scale * X[:, :1]


class TestKnownPositiveSafety:
    """Gate C: identity is not negotiable, whatever the scores say."""

    def _adversarial_data(self):
        # Known positives carry the LOWEST first feature; unlabeled rows are higher.
        rng = np.random.RandomState(3)
        positives = np.full((5, 4), -5.0, dtype=np.float32)
        positives[:, 1:] = rng.normal(0.0, 0.1, (5, 3))
        unlabeled = np.full((20, 4), 2.0, dtype=np.float32)
        unlabeled[:, 1:] = rng.normal(0.0, 0.1, (20, 3))
        X = np.vstack([positives, unlabeled])
        y_pu = np.r_[np.ones(5, dtype=int), np.zeros(20, dtype=int)]
        return X, y_pu

    @pytest.mark.parametrize("view", ["os", "ts"])
    def test_edge_a_known_positive_ranked_lowest_never_becomes_trusted(self, view):
        X, y_pu = self._adversarial_data()
        populations = []
        original_update = self_pu_module.TrustedSetManager.update

        def spy(manager, probabilities, target_size):
            populations.append((manager.n_unlabeled, len(probabilities)))
            return original_update(manager, probabilities, target_size)

        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(self_pu_module.TrustedSetManager, "update", spy)
            model = SelfPUClassifier(**_CONF, backbone=_FirstFeatureBackbone()).fit(
                X, y_pu, os_or_ts=view
            )

        assert populations  # the manager really ran
        assert {n_unlabeled for n_unlabeled, rows in populations} == {20}
        assert {rows for _n_unlabeled, rows in populations} == {20}
        trusted = [index for value in model.trusted_indices_.values() for index in value["indices"]]
        assert trusted  # the arrangement must actually produce a trusted set
        assert set(trusted).isdisjoint(set(np.flatnonzero(y_pu == 1).tolist()))
