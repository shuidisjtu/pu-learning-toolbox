# ruff: noqa: N803, N806

"""UI-independent deployment monitoring and active-review orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from pu_toolbox.diagnostics import (
    PUShiftMonitor,
    PUShiftReport,
    PUUncertaintyReport,
    ShiftWindow,
    analyze_pu_uncertainty,
)


@dataclass(frozen=True)
class DeploymentResult:
    window: ShiftWindow
    shift: PUShiftReport
    review: PUUncertaintyReport
    history: dict[str, Any]


def analyze_deployment_window(
    *,
    reference_X: Any,
    reference_y_pu: Any,
    target_X: Any,
    target_y_pu: Any,
    model: Any,
    window_id: str,
    target_y_true: Any | None = None,
    history_payload: dict[str, Any] | None = None,
    cv: int = 3,
    random_state: int | None = 42,
    min_confidence: float = 0.5,
    query_budget: int = 20,
    query_strategy: str = "uncertainty",
) -> DeploymentResult:
    """Run one target-window audit and review plan without Streamlit imports."""
    monitor = PUShiftMonitor(reference_X, reference_y_pu, cv=cv, random_state=random_state)
    if history_payload is not None:
        monitor.load_history_payload(history_payload)
    window, shift = monitor.update(target_X, y_window_pu=target_y_pu, window_id=window_id)
    review = analyze_pu_uncertainty(
        model,
        target_X,
        y_pu=target_y_pu,
        y_true=target_y_true,
        min_confidence=min_confidence,
        query_budget=query_budget,
        query_strategy=query_strategy,
        random_state=random_state,
    )
    return DeploymentResult(window=window, shift=shift, review=review, history=monitor.to_dict())


def render_deployment_tools(
    st: Any,
    *,
    reference_X: np.ndarray,
    reference_y_pu: np.ndarray,
    model: Any,
) -> None:
    """Render optional target-window monitoring below normal training results."""
    from pu_toolbox.ui.data import load_feature_data, load_label_data

    with st.expander("部署监控与人工复核", expanded=False):
        st.caption("上传一个目标窗口；源训练数据仅作为固定参考域，不会重新训练模型。")
        columns = st.columns(4)
        target_file = columns[0].file_uploader(
            "目标窗口特征", type=["csv", "npy"], key="deployment_target_features"
        )
        target_label_file = columns[1].file_uploader(
            "目标窗口 PU 标签", type=["csv"], key="deployment_target_labels"
        )
        target_truth_file = columns[2].file_uploader(
            "目标窗口真值（可选）", type=["csv"], key="deployment_target_truth"
        )
        history_file = columns[3].file_uploader(
            "既有监控历史（可选）", type=["json"], key="deployment_history"
        )
        settings = st.columns(5)
        window_id = settings[0].text_input("窗口 ID", "current")
        cv = settings[1].number_input("漂移 CV", min_value=2, max_value=10, value=3)
        min_confidence = settings[2].slider("最小置信度", 0.0, 1.0, 0.5, 0.05)
        query_budget = settings[3].number_input("复核预算", min_value=0, value=20)
        strategy = settings[4].selectbox("复核策略", ["uncertainty", "diverse_uncertainty"])
        if target_file is None or target_label_file is None:
            st.info("上传目标窗口特征和 PU 标签后即可运行部署检查。")
            return
        if not st.button("运行部署检查", use_container_width=True):
            return
        try:
            target_X, _ = load_feature_data(target_file.getvalue(), target_file.name)
            target_y = load_label_data(target_label_file.getvalue(), what="target labels")
            target_truth = (
                load_label_data(target_truth_file.getvalue(), what="target truth")
                if target_truth_file is not None
                else None
            )
            history = (
                json.loads(history_file.getvalue().decode("utf-8"))
                if history_file is not None
                else None
            )
            result = analyze_deployment_window(
                reference_X=reference_X,
                reference_y_pu=reference_y_pu,
                target_X=target_X,
                target_y_pu=target_y,
                target_y_true=target_truth,
                model=model,
                window_id=window_id,
                history_payload=history,
                cv=int(cv),
                min_confidence=float(min_confidence),
                query_budget=int(query_budget),
                query_strategy=strategy,
            )
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            st.error(f"部署检查失败：{exc}")
            return
        summary = st.columns(4)
        summary[0].metric("告警级别", result.window.alert_level)
        summary[1].metric("域 AUC", f"{result.window.domain_auc:.3f}")
        summary[2].metric("预测覆盖率", f"{result.review.summary['coverage']:.1%}")
        summary[3].metric("人工复核数", result.review.summary["n_queries"])
        if result.window.alert_codes:
            st.warning("；".join(result.window.alert_codes))
        st.dataframe(
            result.review.to_frame().query("selected_for_review"),
            hide_index=True,
            use_container_width=True,
        )
        downloads = st.columns(3)
        downloads[0].download_button(
            "下载监控历史",
            json.dumps(result.history, ensure_ascii=False, indent=2),
            "shift_history.json",
            "application/json",
        )
        downloads[1].download_button(
            "下载窗口漂移报告", result.shift.to_json(), "window_shift.json", "application/json"
        )
        downloads[2].download_button(
            "下载复核队列",
            result.review.to_frame().to_csv(index=False),
            "uncertainty_rows.csv",
            "text/csv",
        )
