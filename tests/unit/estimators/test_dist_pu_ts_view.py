# ruff: noqa: E402, N803, N806, S101, E501

"""DistPUClassifier 训练视图（协议 §2.3）的数学期望与 OS 冻结基线。

两类证据要分开读：

- **OS 冻结基线**：改造前（`b7d42f2`）在固定 CPU / seed / 小模型下采集的确定性
  数值。结构、长度与有限性**精确断言**；数值**逐 tensor 给容差**——容差吸收的是
  torch/BLAS 版本漂移，不是逻辑改动。敏感性的责任在变异检验与结构断言，
  不在 golden 的松紧。
- **数学期望**：alignment/entropy 的期望值由**独立于生产 helper** 的 numpy 算式
  算出后写成字面量（生成式见本文件末），因此生产实现整体写错时测试仍会翻红。

采集环境：Python 3.11.15 / torch 2.14.0+cu126 / numpy 2.4.6（两次运行逐字节一致，
CPU 路径可复现）。
"""

import inspect
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.risk import DistPUClassifier
from pu_toolbox.estimators.risk import dist_pu as dist_pu_module

# ── 冻结配置（采集与断言共用同一组常量） ────────────────────────────

_GOLDEN_CLASS_PRIOR = 0.35
_GOLDEN_HIDDEN_DIM = 8
_GOLDEN_EPOCHS = 12
_GOLDEN_LEARNING_RATE = 0.01
_GOLDEN_ALIGNMENT_WEIGHT = 1.0
_GOLDEN_ENTROPY_WEIGHT = 0.05
_GOLDEN_RANDOM_STATE = 7

_PROBE = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-1.0, 0.5, 2.0],
        [2.0, -1.0, 0.0],
        [0.5, 0.5, -0.5],
        [-2.0, -2.0, -2.0],
    ],
    dtype=np.float32,
)

# ---- group A: mixup_weight=0.0 ----
_GOLDEN_A_LOSS = [
    0.5956904888153076,
    0.5626175403594971,
    0.5312859416007996,
    0.501765787601471,
    0.4738750457763672,
    0.44802021980285645,
    0.4240368604660034,
    0.40181148052215576,
    0.3809640407562256,
    0.3612164258956909,
    0.34240204095840454,
    0.32442378997802734,
]
_GOLDEN_A_EPOCHS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
_GOLDEN_A_STATE = {
    "0.bias": (
        (8,),
        [
            0.06090084835886955,
            0.020284786820411682,
            0.5264171957969666,
            0.26634305715560913,
            0.4289332628250122,
            -0.32410112023353577,
            0.3135397434234619,
            0.08754612505435944,
        ],
    ),
    "0.weight": (
        (8, 3),
        [
            0.15930995345115662,
            -0.22769424319267273,
            0.3026687502861023,
            0.20850715041160583,
            -0.2697421908378601,
            -0.05269523710012436,
            -0.21884354948997498,
            0.2686198651790619,
            -0.03625711798667908,
            0.5237834453582764,
            0.5282513499259949,
            0.17721132934093475,
            -0.3687981367111206,
            -0.4617549479007721,
            -0.18606893718242645,
            -0.285086989402771,
            0.15008260309696198,
            -0.615676760673523,
            0.44468045234680176,
            -0.5512021780014038,
            0.34401682019233704,
            -0.21219651401042938,
            0.18541653454303741,
            0.1131640076637268,
        ],
    ),
    "2.bias": ((1,), [0.2848155200481415]),
    "2.weight": (
        (1, 8),
        [
            0.045709919184446335,
            -0.06158718839287758,
            0.12714873254299164,
            0.4432840049266815,
            -0.20010040700435638,
            -0.3195667266845703,
            0.02115766704082489,
            -0.1077989712357521,
        ],
    ),
}
_GOLDEN_A_PROBE = [
    0.3827155828475952,
    1.0228406190872192,
    0.4657577872276306,
    0.6237754225730896,
    0.6468194127082825,
    -0.5358678102493286,
]
_GOLDEN_A_TRAIN_HEAD = [
    1.5447360277175903,
    1.1633200645446777,
    1.296680212020874,
    1.2355338335037231,
    1.5667444467544556,
    1.2320822477340698,
]

# ---- group B: mixup_weight=0.3 ----
_GOLDEN_B_LOSS = [
    0.8007653951644897,
    0.7673804759979248,
    0.7352620363235474,
    0.7047152519226074,
    0.6761040687561035,
    0.649402379989624,
    0.6247018575668335,
    0.5993556380271912,
    0.5795371532440186,
    0.5575557351112366,
    0.536942183971405,
    0.5178024768829346,
]
_GOLDEN_B_EPOCHS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
_GOLDEN_B_STATE = {
    "0.bias": (
        (8,),
        [
            0.061351753771305084,
            0.02031334862112999,
            0.5264632105827332,
            0.2663203477859497,
            0.42892301082611084,
            -0.3240794241428375,
            0.3135420083999634,
            0.0875648781657219,
        ],
    ),
    "0.weight": (
        (8, 3),
        [
            0.15937545895576477,
            -0.2276657074689865,
            0.30273866653442383,
            0.20851558446884155,
            -0.269681453704834,
            -0.052645184099674225,
            -0.21886448562145233,
            0.2686006724834442,
            -0.036274537444114685,
            0.5237690210342407,
            0.5282403230667114,
            0.1771983504295349,
            -0.36878377199172974,
            -0.46173620223999023,
            -0.18603026866912842,
            -0.2850959002971649,
            0.14988091588020325,
            -0.615669846534729,
            0.4446909427642822,
            -0.5511938333511353,
            0.34402158856391907,
            -0.21218205988407135,
            0.18543101847171783,
            0.1131773367524147,
        ],
    ),
    "2.bias": ((1,), [0.28481611609458923]),
    "2.weight": (
        (1, 8),
        [
            0.04608004540205002,
            -0.06149366497993469,
            0.1270979791879654,
            0.4432709217071533,
            -0.20006753504276276,
            -0.3195606470108032,
            0.02116013690829277,
            -0.10781237483024597,
        ],
    ),
}
_GOLDEN_B_PROBE = [
    0.38273900747299194,
    1.02289617061615,
    0.46585068106651306,
    0.6240748167037964,
    0.646776020526886,
    -0.5358543395996094,
]
_GOLDEN_B_TRAIN_HEAD = [
    1.5448442697525024,
    1.163364052772522,
    1.2967870235443115,
    1.235694408416748,
    1.5667802095413208,
    1.2321239709854126,
]

# ── 数学期望（独立算式生成，不经过生产 helper） ──────────────────────

_MATH_PROBS = np.array([0.9, 0.7, 0.2, 0.1, 0.4, 0.6])
_MATH_U_MASK = np.array([False, False, True, True, True, True])
_MATH_PI = 0.35
_MATH_OS_ALIGNMENT = 0.0006249999999999984
_MATH_OS_ENTROPY = 0.542875182740249
_MATH_TS_ALIGNMENT = 0.0177777777777778
_MATH_TS_ENTROPY = 0.5179073344025458

_RTOL = 1e-4
_ATOL = 1e-6


def _golden_data():
    """The exact data the golden was captured on (seed 3, 12 P + 28 U)."""
    rng = np.random.RandomState(3)
    positive = rng.normal(1.6, 0.35, size=(12, 3))
    unlabeled = rng.normal(-0.4, 0.9, size=(28, 3))
    X = np.vstack((positive, unlabeled)).astype(np.float32)
    y = np.r_[np.ones(len(positive), dtype=int), np.zeros(len(unlabeled), dtype=int)]
    return X, y


def _fit_golden(mixup_weight, **fit_kwargs):
    """Fit the frozen config.  ``fit_kwargs`` reach ``fit``, so a test can tell
    the default path (no keyword at all) apart from an explicit ``os_or_ts``."""
    model = DistPUClassifier(
        _GOLDEN_CLASS_PRIOR,
        hidden_dim=_GOLDEN_HIDDEN_DIM,
        epochs=_GOLDEN_EPOCHS,
        learning_rate=_GOLDEN_LEARNING_RATE,
        alignment_weight=_GOLDEN_ALIGNMENT_WEIGHT,
        entropy_weight=_GOLDEN_ENTROPY_WEIGHT,
        mixup_weight=mixup_weight,
        random_state=_GOLDEN_RANDOM_STATE,
        device="cpu",
    )
    X, y = _golden_data()
    seen = []
    model.fit(X, y, epoch_callback=lambda epoch, _model: seen.append(epoch), **fit_kwargs)
    return model, X, seen


def _assert_golden(model, X, seen, loss, epochs, state, probe, train_head):
    # 结构：精确
    assert len(model.loss_history_) == _GOLDEN_EPOCHS
    assert np.isfinite(model.loss_history_).all()
    assert seen == epochs

    fitted = {k: v.detach().cpu().numpy() for k, v in model.model_.state_dict().items()}
    assert sorted(fitted) == sorted(state)

    # 数值：逐 tensor 各自设容差
    np.testing.assert_allclose(model.loss_history_, loss, rtol=_RTOL, atol=_ATOL)
    for name, (shape, values) in state.items():
        assert fitted[name].shape == tuple(shape)
        np.testing.assert_allclose(fitted[name].ravel(), values, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(model._decision_function(_PROBE), probe, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(model._decision_function(X)[:6], train_head, rtol=_RTOL, atol=_ATOL)


# ── OS 冻结基线 ────────────────────────────────────────────────────


@pytest.mark.unit
def test_determ_os_golden_matches_the_pre_refactor_run_without_mixup():
    """Group A: `mixup_weight=0`, so `randperm`/`rand` never run."""
    model, X, seen = _fit_golden(0.0)

    _assert_golden(
        model,
        X,
        seen,
        _GOLDEN_A_LOSS,
        _GOLDEN_A_EPOCHS,
        _GOLDEN_A_STATE,
        _GOLDEN_A_PROBE,
        _GOLDEN_A_TRAIN_HEAD,
    )


@pytest.mark.unit
def test_determ_os_golden_matches_the_pre_refactor_run_with_mixup():
    """Group B: the Mixup branch is live, so the RNG stream is in play."""
    model, X, seen = _fit_golden(0.3)

    _assert_golden(
        model,
        X,
        seen,
        _GOLDEN_B_LOSS,
        _GOLDEN_B_EPOCHS,
        _GOLDEN_B_STATE,
        _GOLDEN_B_PROBE,
        _GOLDEN_B_TRAIN_HEAD,
    )


@pytest.mark.unit
def test_basic_default_view_equals_explicit_os():
    """Self-consistency, and *not* a substitute for the golden above: it proves
    only that the new flag does nothing extra on the default path."""
    default, _, _ = _fit_golden(0.0)
    explicit, _, _ = _fit_golden(0.0, os_or_ts="os")

    assert default.loss_history_ == explicit.loss_history_


# ── 数学期望：两视图的角色集合 ─────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "alignment", "entropy"),
    [
        ("os", _MATH_OS_ALIGNMENT, _MATH_OS_ENTROPY),
        ("ts", _MATH_TS_ALIGNMENT, _MATH_TS_ENTROPY),
    ],
)
@pytest.mark.unit
def test_basic_regularizers_match_independently_computed_literals(path, alignment, entropy):
    get_alignment, get_entropy = dist_pu_module._distribution_regularizers(
        torch.tensor(_MATH_PROBS, dtype=torch.float64),
        torch.tensor(_MATH_U_MASK),
        _MATH_PI,
        include_positive_in_unlabeled=(path == "ts"),
    )

    assert float(get_alignment) == pytest.approx(alignment, rel=1e-9, abs=1e-15)
    assert float(get_entropy) == pytest.approx(entropy, rel=1e-9, abs=1e-15)


@pytest.mark.unit
def test_edge_the_ts_role_set_is_the_whole_training_matrix():
    """Recover each view's role mean from its alignment and compare it with the
    mean over the row set the protocol says that view must use."""
    os_alignment, _ = dist_pu_module._distribution_regularizers(
        torch.tensor(_MATH_PROBS, dtype=torch.float64),
        torch.tensor(_MATH_U_MASK),
        _MATH_PI,
        include_positive_in_unlabeled=False,
    )
    ts_alignment, _ = dist_pu_module._distribution_regularizers(
        torch.tensor(_MATH_PROBS, dtype=torch.float64),
        torch.tensor(_MATH_U_MASK),
        _MATH_PI,
        include_positive_in_unlabeled=True,
    )

    os_mean = _MATH_PI - math.sqrt(float(os_alignment))
    ts_mean = _MATH_PI + math.sqrt(float(ts_alignment))

    assert os_mean == pytest.approx(float(_MATH_PROBS[_MATH_U_MASK].mean()), rel=1e-9)
    assert ts_mean == pytest.approx(float(_MATH_PROBS.mean()), rel=1e-9)
    # 2 positive rows on top of the 4 unlabeled ones -- the role set grew by
    # exactly the labeled positives, and nothing was duplicated.
    assert _MATH_U_MASK.sum() == 4
    assert len(_MATH_PROBS) - _MATH_U_MASK.sum() == 2


@pytest.mark.unit
def test_basic_alignment_target_is_the_same_population_prior_in_both_views():
    """A U-role whose mean equals π must give exactly zero alignment under OS.

    If the target were re-derived from the observed P/U ratio, this would not
    hold; and the TS side must not collapse to zero just because the OS side
    does, which is what makes the role set difference visible.
    """
    probs = np.array([0.9, 0.7, 0.5, 0.3])
    mask = np.array([False, False, True, True])
    pi = 0.4  # equals mean(probs[mask]) but not mean(probs) == 0.6

    os_alignment, _ = dist_pu_module._distribution_regularizers(
        torch.tensor(probs, dtype=torch.float64),
        torch.tensor(mask),
        pi,
        include_positive_in_unlabeled=False,
    )
    ts_alignment, _ = dist_pu_module._distribution_regularizers(
        torch.tensor(probs, dtype=torch.float64),
        torch.tensor(mask),
        pi,
        include_positive_in_unlabeled=True,
    )

    assert float(os_alignment) == pytest.approx(0.0, abs=1e-15)
    assert float(ts_alignment) == pytest.approx((0.6 - 0.4) ** 2, rel=1e-9)


# ── 视图 API ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_basic_ts_view_fits_and_moves_the_solution():
    """Not a no-op: the calibrated view fits a different model."""
    os_model, X, _ = _fit_golden(0.0)
    ts_model, _, _ = _fit_golden(0.0, os_or_ts="ts")

    assert ts_model.loss_history_ != os_model.loss_history_
    assert not np.array_equal(ts_model._decision_function(X), os_model._decision_function(X))
    assert ts_model.predict(X).shape == (len(X),)


@pytest.mark.parametrize("bad", ["TS", ""])
@pytest.mark.unit
def test_param_invalid_view_value_is_rejected(bad):
    X, y = _golden_data()
    with pytest.raises(ValueError, match="os_or_ts must be"):
        DistPUClassifier(_GOLDEN_CLASS_PRIOR, hidden_dim=4, epochs=1).fit(X, y, os_or_ts=bad)


@pytest.mark.unit
def test_param_the_view_flag_is_keyword_only_and_defaults_to_os():
    params = inspect.signature(DistPUClassifier.fit).parameters

    assert params["os_or_ts"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["os_or_ts"].default == "os"


# 数学字面量的生成式（独立于生产实现，可复算）：
#
#   probs = np.array([0.9, 0.7, 0.2, 0.1, 0.4, 0.6]); u = np.array([0,0,1,1,1,1]); pi = 0.35
#   def reg(role):
#       return (float((role.mean() - pi) ** 2),
#               float(-(role*np.log(role+1e-6) + (1-role)*np.log(1-role+1e-6)).mean()))
#   reg(probs[u])  -> (0.0006249999999999984, 0.542875182740249)
#   reg(probs)     -> (0.0177777777777778,  0.5179073344025458)
