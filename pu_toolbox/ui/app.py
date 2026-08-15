# ruff: noqa: N803, N806

"""Streamlit application and dependency-light UI helpers."""

from __future__ import annotations

import inspect
import io
import json
import pickle
from typing import Any

import numpy as np
import pandas as pd

from pu_toolbox.core.base import BasePUClassifier
from pu_toolbox.model_selection.tuning import PUTuner, TuningResult
from pu_toolbox.registry import get_algorithm, list_algorithms, register_all_builtin_methods
from pu_toolbox.workflows import DEFAULT_METRICS, PUPipeline

_MANAGED_PARAMS = {"class_prior", "random_state", "encoder"}


def parse_json_mapping(text: str, *, field_name: str) -> dict[str, Any]:
    """Parse a JSON object used by the advanced parameter editors."""
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{field_name} is not valid JSON: line {exc.lineno}, column {exc.colno}."
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"{field_name} keys must be non-empty strings.")
    return value


def _read_csv_bytes(content: bytes, *, what: str) -> pd.DataFrame:
    if not content:
        raise ValueError(f"{what} file is empty.")
    frame = pd.read_csv(io.BytesIO(content))
    if frame.empty:
        raise ValueError(f"{what} file has no data rows.")
    # A fully numeric first row is consumed as column names by pandas. Keep
    # the CLI's safety contract and reject this ambiguous, headerless input.
    try:
        [float(column) for column in frame.columns]
    except (TypeError, ValueError):
        pass
    else:
        raise ValueError(f"{what} CSV needs a non-numeric header row.")
    return frame


def load_feature_data(content: bytes, filename: str) -> tuple[np.ndarray, list[str]]:
    """Load a UI upload as numeric CSV table or 4-D NCHW ``.npy`` data."""
    if filename.lower().endswith(".npy"):
        array = np.load(io.BytesIO(content), allow_pickle=False)
        if array.ndim != 4:
            raise ValueError(f"image data must be 4-D NCHW; got shape {array.shape}.")
        array = array.astype(np.float32, copy=False)
        columns = [f"channel_{index}" for index in range(array.shape[1])]
    else:
        frame = _read_csv_bytes(content, what="feature")
        try:
            array = frame.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("feature CSV must contain only numeric values.") from exc
        columns = [str(column) for column in frame.columns]
    if not np.isfinite(array).all():
        raise ValueError("feature data contains NaN or Inf values; clean or impute it first.")
    return array, columns


def load_label_data(content: bytes, *, what: str = "labels") -> np.ndarray:
    """Load a single-column CSV label upload."""
    frame = _read_csv_bytes(content, what=what)
    if frame.shape[1] != 1:
        raise ValueError(f"{what} CSV must have exactly one column; got {frame.shape[1]}.")
    try:
        values = frame.iloc[:, 0].to_numpy(dtype=int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must contain integer labels.") from exc
    return values


def _annotation_text(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "unspecified"
    return str(annotation).replace("typing.", "")


def classifier_catalog() -> list[dict[str, Any]]:
    """Return trainable classifier metadata and constructor fields for the UI."""
    register_all_builtin_methods()
    catalog: list[dict[str, Any]] = []
    for metadata in sorted(list_algorithms(trainable_only=True), key=lambda item: item.name):
        try:
            cls = get_algorithm(metadata.name)
        except Exception:  # noqa: BLE001 - registry may hold unavailable optional implementations
            continue
        if not isinstance(cls, type) or not issubclass(cls, BasePUClassifier):
            continue
        parameters = []
        for name, parameter in inspect.signature(cls.__init__).parameters.items():
            if name == "self" or name in _MANAGED_PARAMS:
                continue
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            required = parameter.default is inspect.Parameter.empty
            parameters.append(
                {
                    "name": name,
                    "type": _annotation_text(parameter.annotation),
                    "default": None if required else repr(parameter.default),
                    "required": required,
                }
            )
        ui_ready = all(
            not parameter["required"] or parameter["type"] in {"bool", "float", "int", "str"}
            for parameter in parameters
        )
        catalog.append(
            {
                "name": metadata.name,
                "family": metadata.family.value,
                "requires_class_prior": metadata.requires_class_prior,
                "supports_gpu": metadata.supports_gpu,
                "ui_ready": ui_ready,
                "parameters": parameters,
            }
        )
    return catalog


def _metric_rows(report: Any) -> list[dict[str, Any]]:
    return [
        {
            "metric": name,
            "mean": metric.mean,
            "std": metric.std,
            "basis": metric.basis,
            "available": metric.available,
        }
        for name, metric in report.cv_metrics.items()
    ]


def _downloads(st: Any, report: Any, tuning: TuningResult | None, X: np.ndarray) -> None:
    predictions = report.final_model.predict(X)
    prediction_csv = pd.DataFrame({"prediction": predictions}).to_csv(index=False)
    columns = st.columns(4)
    columns[0].download_button(
        "下载 JSON 报告",
        report.to_json(),
        "report.json",
        "application/json",
        use_container_width=True,
    )
    columns[1].download_button(
        "下载 Markdown",
        report.to_markdown(),
        "report.md",
        "text/markdown",
        use_container_width=True,
    )
    columns[2].download_button(
        "下载预测结果",
        prediction_csv,
        "predictions.csv",
        "text/csv",
        use_container_width=True,
    )
    try:
        model_bytes = pickle.dumps(report.final_model)
    except Exception:  # noqa: BLE001 - some user-injected torch modules are not picklable
        columns[3].button("模型不可序列化", disabled=True, use_container_width=True)
    else:
        columns[3].download_button(
            "下载训练模型",
            model_bytes,
            "model.pkl",
            "application/octet-stream",
            use_container_width=True,
        )
    if tuning is not None:
        st.download_button(
            "下载调参记录",
            json.dumps(tuning.to_dict(), ensure_ascii=False, indent=2),
            "tuning.json",
            "application/json",
        )


def main() -> None:
    """Render the application. Streamlit is imported only for this entry point."""
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - launcher provides the normal message
        raise RuntimeError('Install UI dependencies with: pip install "pu-toolbox[ui]"') from exc

    st.set_page_config(page_title="PU Learning Toolbox", page_icon="🧰", layout="wide")
    st.title("PU Learning Toolbox")
    st.caption("上传数据、选择或调整模型，并在一个页面内完成 PU 训练、诊断与结果下载。")

    st.subheader("1 · 上传数据")
    upload_columns = st.columns(3)
    feature_file = upload_columns[0].file_uploader("特征数据", type=["csv", "npy"])
    label_file = upload_columns[1].file_uploader("PU 标签（1/0 单列 CSV）", type=["csv"])
    truth_file = upload_columns[2].file_uploader("真实标签（可选，1/0 单列 CSV）", type=["csv"])
    if feature_file is None or label_file is None:
        st.info("请先上传特征数据和 PU 标签。CSV 第一行必须是非数字列名。")
        return

    try:
        X, feature_names = load_feature_data(feature_file.getvalue(), feature_file.name)
        y_pu = load_label_data(label_file.getvalue())
        y_true = (
            load_label_data(truth_file.getvalue(), what="true labels")
            if truth_file is not None
            else None
        )
        if len(y_pu) != X.shape[0]:
            raise ValueError(f"feature rows ({X.shape[0]}) and labels ({len(y_pu)}) differ.")
        if y_true is not None and len(y_true) != X.shape[0]:
            raise ValueError(f"feature rows ({X.shape[0]}) and true labels ({len(y_true)}) differ.")
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return

    summary_columns = st.columns(4)
    summary_columns[0].metric("样本数", X.shape[0])
    summary_columns[1].metric("特征数", int(np.prod(X.shape[1:])))
    summary_columns[2].metric("已标正例", int(np.sum(y_pu == 1)))
    summary_columns[3].metric("未标记样本", int(np.sum(y_pu == 0)))
    with st.expander("数据预览", expanded=False):
        if X.ndim == 2:
            st.dataframe(pd.DataFrame(X[:100], columns=feature_names), use_container_width=True)
        else:
            st.write(f"图像数组形状：{X.shape}（NCHW）")

    st.subheader("2 · 配置模型")
    catalog = classifier_catalog()
    catalog_by_name = {item["name"]: item for item in catalog}
    config_columns = st.columns(3)
    image_mode = X.ndim == 4
    selection_options = ["手动选择"] if image_mode else ["自动推荐", "手动选择"]
    selection_mode = config_columns[0].radio("选择方式", selection_options, horizontal=True)
    classifier = "auto"
    if selection_mode == "手动选择":
        choices = sorted(
            name
            for name, item in catalog_by_name.items()
            if item["ui_ready"]
            and (not image_mode or name in {"infomax_pu", "weighted_contrastive_pu"})
        )
        classifier = config_columns[1].selectbox("分类器", choices)
        selected = catalog_by_name[classifier]
        config_columns[2].markdown(
            f"族：`{selected['family']}`  \n"
            f"需要类先验：`{selected['requires_class_prior']}`  \n"
            f"GPU：`{selected['supports_gpu']}`"
        )
    if image_mode:
        st.info("图像输入使用 CNN 模式，目前支持 InfoMax PU 与 WConPU。")

    settings = st.columns(4)
    cv = settings[0].number_input("交叉验证折数", min_value=2, max_value=20, value=5)
    seed = settings[1].number_input("随机种子", min_value=0, value=42)
    prior_method = settings[2].selectbox("类先验", ["自动估计", "手动输入", "不使用"])
    prior_estimator = settings[3].selectbox(
        "先验估计器", ["pen_l1", "recpe", "km1", "km2"], disabled=prior_method != "自动估计"
    )
    class_prior = None
    if prior_method == "手动输入":
        class_prior = st.slider("正类比例 π", 0.01, 0.99, 0.50, 0.01)
    elif prior_method == "不使用":
        prior_estimator = None

    metric_options = list(DEFAULT_METRICS) + ["pu_accuracy", "pu_f1", "pu_negative_rate"]
    metrics = st.multiselect("评估指标", metric_options, default=list(DEFAULT_METRICS))
    if not metrics:
        st.warning("至少选择一个评估指标。")

    classifier_params: dict[str, Any] = {}
    tuning_enabled = False
    tuning_grid: dict[str, Any] = {}
    scoring = "pu_zero_one_risk"
    backbone = "cnn13"
    if image_mode:
        backbone = st.selectbox("CNN 骨架", ["cnn13", "resnet18", "resnet50"])
    if classifier != "auto":
        with st.expander("模型参数与调参", expanded=True):
            parameter_rows = catalog_by_name[classifier]["parameters"]
            if parameter_rows:
                st.dataframe(parameter_rows, hide_index=True, use_container_width=True)
            parameter_text = st.text_area(
                "固定参数（JSON 对象）",
                "{}",
                help='例如：{"reg_lambda": 0.01, "max_iter": 1000}',
            )
            tuning_enabled = st.toggle("比较多组超参数")
            if tuning_enabled:
                tuning_text = st.text_area(
                    "参数网格（每个值必须是列表）",
                    "{}",
                    help='例如：{"reg_lambda": [0.001, 0.01, 0.1]}',
                )
                scoring = st.selectbox("选择最佳模型的指标", metrics or metric_options)
            try:
                classifier_params = parse_json_mapping(
                    parameter_text, field_name="fixed parameters"
                )
                if tuning_enabled:
                    tuning_grid = parse_json_mapping(tuning_text, field_name="parameter grid")
                    if not tuning_grid:
                        raise ValueError("parameter grid cannot be empty when tuning is enabled.")
                    tuning_grid = {
                        key: value if isinstance(value, list) else [value]
                        for key, value in tuning_grid.items()
                    }
                    overlap = classifier_params.keys() & tuning_grid.keys()
                    if overlap:
                        raise ValueError(
                            f"parameters cannot be both fixed and tuned: {sorted(overlap)}"
                        )
                    tuning_grid = {
                        **{key: [value] for key, value in classifier_params.items()},
                        **tuning_grid,
                    }
            except ValueError as exc:
                st.error(str(exc))
                return

    st.subheader("3 · 训练与结果")
    if not st.button("开始分析", type="primary", use_container_width=True, disabled=not metrics):
        return

    common = {
        "prior_estimator": prior_estimator,
        "cv": int(cv),
        "metrics": metrics,
        "random_state": int(seed),
        "architecture": "cnn" if X.ndim == 4 else "mlp",
        "backbone": backbone,
        "device": "auto",
    }
    try:
        with st.spinner("正在执行数据画像、PU 分层验证和模型训练……"):
            tuning = None
            if tuning_enabled:
                tuner = PUTuner(
                    classifier=classifier,
                    param_grid=tuning_grid,
                    scoring=scoring,
                    **common,
                )
                tuning = tuner.fit(X, y_pu, y_true=y_true, class_prior=class_prior)
                report = tuning.best_report
            else:
                report = PUPipeline(
                    classifier=classifier,
                    classifier_params=classifier_params,
                    **common,
                ).fit_evaluate(X, y_pu, y_true=y_true, class_prior=class_prior)
    except Exception as exc:  # noqa: BLE001 - UI must turn backend failures into readable feedback
        st.error(f"分析失败：{exc}")
        return

    st.success("分析完成。")
    if tuning is not None:
        st.write("最佳参数", tuning.best_params)
        st.metric(f"最佳 {tuning.scoring}", f"{tuning.best_score:.6f}")
        st.dataframe([trial.to_dict() for trial in tuning.trials], use_container_width=True)

    metric_frame = pd.DataFrame(_metric_rows(report))
    st.dataframe(metric_frame, hide_index=True, use_container_width=True)
    chart = metric_frame.loc[metric_frame["available"], ["metric", "mean"]].set_index("metric")
    if not chart.empty:
        st.bar_chart(chart)

    st.markdown("#### 诊断提示")
    if not report.issues:
        st.success("当前检查未发现问题。")
    for issue in report.issues:
        message = f"**{issue.code}**：{issue.message}  \n建议：{issue.action}"
        if issue.severity == "error":
            st.error(message)
        elif issue.severity == "warning":
            st.warning(message)
        else:
            st.info(message)
    _downloads(st, report, tuning, X)


if __name__ == "__main__":
    main()
