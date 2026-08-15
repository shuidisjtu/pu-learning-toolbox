# ruff: noqa: N803, N806

"""Streamlit page coordinator for PU Learning Toolbox."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from pu_toolbox.run_config import RunConfiguration
from pu_toolbox.ui.configuration import apply_run_configuration, parse_json_mapping
from pu_toolbox.ui.data import load_feature_data, load_label_data
from pu_toolbox.ui.execution import AnalysisResult, execute_analysis
from pu_toolbox.ui.parameters import classifier_catalog, render_parameter_form
from pu_toolbox.ui.results import render_results
from pu_toolbox.ui.runtime import BackgroundRun, submit_background
from pu_toolbox.workflows import DEFAULT_METRICS


def main() -> None:
    """Render the application. Streamlit is imported only for this entry point."""
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - launcher provides the normal message
        raise RuntimeError('Install UI dependencies with: pip install "pu-toolbox[ui]"') from exc

    st.set_page_config(page_title="PU Learning Toolbox", page_icon="🧰", layout="wide")
    st.title("PU Learning Toolbox")
    st.caption("上传数据、选择或调整模型，并在一个页面内完成 PU 训练、诊断与结果下载。")

    with st.sidebar:
        st.subheader("运行配置")
        config_file = st.file_uploader("导入配置（可选）", type=["json"], key="run_config_file")
        imported_config = None
        imported_digest = None
        if config_file is not None:
            try:
                config_bytes = config_file.getvalue()
                imported_config = RunConfiguration.from_json(config_bytes.decode("utf-8"))
                imported_digest = sha256(config_bytes).hexdigest()
            except (UnicodeDecodeError, ValueError) as exc:
                st.error(f"配置导入失败：{exc}")
        history = st.session_state.get("run_history", [])
        with st.expander(f"运行历史（{len(history)}）", expanded=False):
            if history:
                st.dataframe(history, hide_index=True, use_container_width=True)
            else:
                st.caption("当前浏览器会话还没有运行记录。")

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
    metric_options = list(DEFAULT_METRICS) + ["pu_accuracy", "pu_f1", "pu_negative_rate"]
    if imported_config is not None and imported_digest != st.session_state.get(
        "applied_config_digest"
    ):
        try:
            if (
                imported_config.classifier != "auto"
                and imported_config.classifier not in catalog_by_name
            ):
                raise ValueError(
                    f"classifier {imported_config.classifier!r} is not available in this install."
                )
            unavailable_comparisons = sorted(
                set(imported_config.comparison_classifiers) - set(catalog_by_name)
            )
            if unavailable_comparisons:
                raise ValueError(
                    f"comparison classifiers are not available: {unavailable_comparisons}."
                )
            ineligible_comparisons = [
                name
                for name in imported_config.comparison_classifiers
                if any(parameter["required"] for parameter in catalog_by_name[name]["parameters"])
                or (X.ndim == 4 and name not in {"infomax_pu", "weighted_contrastive_pu"})
            ]
            if ineligible_comparisons:
                raise ValueError(
                    "comparison classifiers need unsupported required parameters or input mode: "
                    f"{ineligible_comparisons}."
                )
            unknown_metrics = sorted(set(imported_config.metrics) - set(metric_options))
            if unknown_metrics:
                raise ValueError(f"unsupported UI metrics: {unknown_metrics}.")
            if imported_config.architecture != ("cnn" if X.ndim == 4 else "mlp"):
                raise ValueError(
                    "configuration architecture does not match the uploaded feature data."
                )
            apply_run_configuration(st.session_state, imported_config, catalog_by_name)
            st.session_state["applied_config_digest"] = imported_digest
            st.sidebar.success("配置已应用。")
        except ValueError as exc:
            st.sidebar.error(f"配置无法应用：{exc}")
    config_columns = st.columns(3)
    image_mode = X.ndim == 4
    selection_options = (
        ["手动选择", "比较模型"] if image_mode else ["自动推荐", "手动选择", "比较模型"]
    )
    selection_mode = config_columns[0].radio(
        "选择方式", selection_options, horizontal=True, key="selection_mode"
    )
    classifier = "auto"
    comparison_classifiers: list[str] = []
    if selection_mode == "手动选择":
        choices = sorted(
            name
            for name, item in catalog_by_name.items()
            if item["ui_ready"]
            and (not image_mode or name in {"infomax_pu", "weighted_contrastive_pu"})
        )
        classifier = config_columns[1].selectbox("分类器", choices, key="classifier")
        selected = catalog_by_name[classifier]
        config_columns[2].markdown(
            f"族：`{selected['family']}`  \n"
            f"需要类先验：`{selected['requires_class_prior']}`  \n"
            f"GPU：`{selected['supports_gpu']}`"
        )
    elif selection_mode == "比较模型":
        choices = sorted(
            name
            for name, item in catalog_by_name.items()
            if item["ui_ready"]
            and not any(parameter["required"] for parameter in item["parameters"])
            and (not image_mode or name in {"infomax_pu", "weighted_contrastive_pu"})
        )
        comparison_classifiers = config_columns[1].multiselect(
            "待比较模型",
            choices,
            default=choices[:2],
            key="comparison_classifiers",
        )
        config_columns[2].caption("所有模型使用相同的先验、CV、指标和随机种子。")
    if image_mode:
        st.info("图像输入使用 CNN 模式，目前支持 InfoMax PU 与 WConPU。")

    settings = st.columns(4)
    cv = settings[0].number_input("交叉验证折数", min_value=2, max_value=20, value=5, key="cv")
    seed = settings[1].number_input("随机种子", min_value=0, value=42, key="seed")
    prior_method = settings[2].selectbox(
        "类先验", ["自动估计", "手动输入", "不使用"], key="prior_method"
    )
    prior_estimator = settings[3].selectbox(
        "先验估计器",
        ["pen_l1", "recpe", "km1", "km2"],
        disabled=prior_method != "自动估计",
        key="prior_estimator",
    )
    class_prior = None
    if prior_method == "手动输入":
        class_prior = st.slider("正类比例 π", 0.01, 0.99, 0.50, 0.01, key="class_prior")
    elif prior_method == "不使用":
        prior_estimator = None

    metrics = st.multiselect(
        "评估指标", metric_options, default=list(DEFAULT_METRICS), key="metrics"
    )
    if not metrics:
        st.warning("至少选择一个评估指标。")
    if comparison_classifiers:
        scoring = st.selectbox(
            "选择最佳模型的指标", metrics or metric_options, key="comparison_scoring"
        )
    else:
        scoring = "pu_zero_one_risk"

    classifier_params: dict[str, Any] = {}
    tuning_enabled = False
    tuning_grid: dict[str, Any] = {}
    configuration_grid: dict[str, Any] = {}
    backbone = "cnn13"
    if image_mode:
        backbone = st.selectbox("CNN 骨架", ["cnn13", "resnet18", "resnet50"], key="backbone")
    if classifier != "auto":
        with st.expander("模型参数与调参", expanded=True):
            parameter_rows = catalog_by_name[classifier]["parameters"]
            if parameter_rows:
                st.dataframe(
                    [
                        {
                            "参数": item["name"],
                            "类型": item["type"],
                            "默认值": item["default"],
                            "必填": item["required"],
                        }
                        for item in parameter_rows
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
            classifier_params = render_parameter_form(st, classifier, parameter_rows)
            parameter_text = st.text_area(
                "额外固定参数（高级 JSON）",
                "{}",
                help="用于复杂对象或批量粘贴；不能与上方已选择的参数重名。",
                key="parameter_text",
            )
            tuning_enabled = st.toggle("比较多组超参数", key="tuning_enabled")
            if tuning_enabled:
                tuning_text = st.text_area(
                    "参数网格（每个值必须是列表）",
                    "{}",
                    help='例如：{"reg_lambda": [0.001, 0.01, 0.1]}',
                    key="tuning_text",
                )
                scoring = st.selectbox(
                    "选择最佳模型的指标", metrics or metric_options, key="scoring"
                )
            try:
                advanced_params = parse_json_mapping(parameter_text, field_name="fixed parameters")
                overlap = classifier_params.keys() & advanced_params.keys()
                if overlap:
                    raise ValueError(
                        f"parameters cannot be set in both typed fields and JSON: {sorted(overlap)}"
                    )
                classifier_params = {**classifier_params, **advanced_params}
                if tuning_enabled:
                    configuration_grid = parse_json_mapping(
                        tuning_text, field_name="parameter grid"
                    )
                    if not configuration_grid:
                        raise ValueError("parameter grid cannot be empty when tuning is enabled.")
                    configuration_grid = {
                        key: value if isinstance(value, list) else [value]
                        for key, value in configuration_grid.items()
                    }
                    overlap = classifier_params.keys() & configuration_grid.keys()
                    if overlap:
                        raise ValueError(
                            f"parameters cannot be both fixed and tuned: {sorted(overlap)}"
                        )
                    tuning_grid = {
                        **{key: [value] for key, value in classifier_params.items()},
                        **configuration_grid,
                    }
            except ValueError as exc:
                st.error(str(exc))
                return

    run_configuration = RunConfiguration(
        classifier=classifier,
        classifier_params=classifier_params,
        prior_estimator=prior_estimator,
        class_prior=class_prior,
        cv=int(cv),
        metrics=tuple(metrics),
        random_state=int(seed),
        architecture="cnn" if X.ndim == 4 else "mlp",
        backbone=backbone,
        device="auto",
        tuning_grid=configuration_grid,
        scoring=scoring,
        comparison_classifiers=tuple(comparison_classifiers),
    )
    st.download_button(
        "导出运行配置",
        run_configuration.to_json(),
        "pu-run-config.json",
        "application/json",
        disabled=not metrics,
    )

    st.subheader("3 · 训练与结果")
    invalid_comparison = selection_mode == "比较模型" and len(comparison_classifiers) < 2
    if invalid_comparison:
        st.warning("模型比较至少需要选择两个模型。")
    active_run: BackgroundRun | None = st.session_state.get("active_run")
    start_clicked = st.button(
        "开始分析",
        type="primary",
        use_container_width=True,
        disabled=not metrics or invalid_comparison or active_run is not None,
    )

    common = {
        "prior_estimator": prior_estimator,
        "cv": int(cv),
        "metrics": metrics,
        "random_state": int(seed),
        "architecture": "cnn" if X.ndim == 4 else "mlp",
        "backbone": backbone,
        "device": "auto",
    }
    if start_clicked:
        run_mode = (
            "comparison" if comparison_classifiers else "tuning" if tuning_enabled else "pipeline"
        )

        def task(token, callback):
            return execute_analysis(
                X=X,
                y_pu=y_pu,
                y_true=y_true,
                class_prior=class_prior,
                classifier=classifier,
                classifier_params=classifier_params,
                tuning_grid=tuning_grid,
                comparison_classifiers=comparison_classifiers,
                scoring=scoring,
                pipeline_params=common,
                cancellation_token=token,
                progress_callback=callback,
            )

        active_run = submit_background(task)
        st.session_state["active_run"] = active_run
        st.session_state["active_run_mode"] = run_mode
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_error", None)

    if active_run is not None:
        snapshot = active_run.snapshot()
        st.progress(snapshot.progress.fraction, text=snapshot.progress.message)
        status_columns = st.columns([3, 1])
        status_columns[0].caption(
            f"阶段：{snapshot.progress.stage} · 状态：{snapshot.status} · "
            f"开始：{snapshot.started_at}"
        )
        if status_columns[1].button(
            "取消运行",
            disabled=snapshot.status in {"cancelled", "completed", "failed"},
            use_container_width=True,
        ):
            active_run.cancel()

        if active_run.future.done():
            mode = st.session_state.pop("active_run_mode", "pipeline")
            history_entry = {
                "开始时间": snapshot.started_at,
                "结束时间": datetime.now(UTC).isoformat(),
                "模式": mode,
            }
            try:
                analysis = active_run.future.result()
            except Exception as exc:  # noqa: BLE001 - UI error boundary
                status = "cancelled" if active_run.token.is_cancelled else "failed"
                history_entry["状态"] = status
                history_entry["结果"] = str(exc) or "run cancelled by user"
                st.session_state["analysis_error"] = history_entry["结果"]
            else:
                history_entry["状态"] = "completed"
                history_entry["结果"] = analysis.report.provenance.get("classifier", "unknown")
                st.session_state["analysis_result"] = analysis
            history = [history_entry, *st.session_state.get("run_history", [])][:20]
            st.session_state["run_history"] = history
            st.session_state.pop("active_run", None)
            st.rerun()
        time.sleep(0.4)
        st.rerun()
        return

    error = st.session_state.get("analysis_error")
    if error:
        st.error(f"分析失败：{error}")
        return
    analysis: AnalysisResult | None = st.session_state.get("analysis_result")
    if analysis is None:
        return
    render_results(st, analysis, X)


if __name__ == "__main__":
    main()
