# ruff: noqa: N803

"""Streamlit result tables, diagnostics, and downloads."""

from __future__ import annotations

import json
import pickle
from typing import Any

import numpy as np
import pandas as pd

from pu_toolbox.ui.deployment import render_deployment_tools
from pu_toolbox.ui.execution import AnalysisResult


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


def render_results(st: Any, analysis: AnalysisResult, X: np.ndarray, y_pu: np.ndarray) -> None:
    """Render one completed analysis and all its downloadable artifacts."""
    report = analysis.report
    tuning = analysis.tuning
    comparison = analysis.comparison
    st.success("分析完成。")
    if comparison is not None:
        st.write("最佳模型", comparison.best_classifier)
        st.metric(f"最佳 {comparison.scoring}", f"{comparison.best_score:.6f}")
        st.dataframe([trial.to_dict() for trial in comparison.trials], use_container_width=True)
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
    _downloads(st, analysis, X)
    render_deployment_tools(
        st,
        reference_X=X,
        reference_y_pu=y_pu,
        model=report.final_model,
    )


def _downloads(st: Any, analysis: AnalysisResult, X: np.ndarray) -> None:
    report = analysis.report
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
    if analysis.tuning is not None:
        st.download_button(
            "下载调参记录",
            json.dumps(analysis.tuning.to_dict(), ensure_ascii=False, indent=2),
            "tuning.json",
            "application/json",
        )
    if analysis.comparison is not None:
        st.download_button(
            "下载模型比较记录",
            json.dumps(analysis.comparison.to_dict(), ensure_ascii=False, indent=2),
            "comparison.json",
            "application/json",
        )
