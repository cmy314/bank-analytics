"""
加载 if_v2021.joblib，对 Prometheus 导出 CSV 做四维 IF + P95 推理。
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import pandas as pd

from bank_analytics.prometheus_csv import normalize_prometheus_csv
from bank_analytics.v2021_model import _apply_p95_rule_alerts


def load_if_artifact(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    artifact = joblib.load(model_path)
    required = ("model", "scaler", "feat_cols")
    missing = [k for k in required if k not in artifact]
    if missing:
        raise ValueError(f"joblib 缺少字段: {missing}")
    return artifact


def _align_feature_frame(
    norm_df: pd.DataFrame, artifact: dict[str, Any]
) -> tuple[pd.DataFrame, str, list[str]]:
    feat_cols: list[str] = list(artifact["feat_cols"])
    rt_key = artifact.get("rt_key") or next(
        (c for c in feat_cols if str(c).endswith("_RT") or c in ("rt", "HTTP_RT")),
        "rt",
    )

    work = norm_df.copy()
    if rt_key != "rt" and rt_key not in work.columns:
        work[rt_key] = work["rt"]

    for col in feat_cols:
        if col not in work.columns:
            raise ValueError(
                f"推理宽表缺少模型列 {col}，已有: {work.columns.tolist()}。"
                "请确认 joblib 为四维 QPS/RT/CPU/Memory 训练产物。"
            )
    work = work.dropna(subset=feat_cols).reset_index(drop=True)
    return work, rt_key, feat_cols


def infer_alerts_from_csv(
    csv_path: str | os.PathLike[str],
    model_path: str | os.PathLike[str],
    *,
    rt_unit: str = "auto",
    cpu_unit: str = "auto",
    memory_unit: str = "auto",
) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, low_memory=False)
    norm = normalize_prometheus_csv(
        raw,
        rt_unit=rt_unit,
        cpu_unit=cpu_unit,
        memory_unit=memory_unit,
    )
    artifact = load_if_artifact(model_path)
    work, rt_key, feat_cols = _align_feature_frame(norm, artifact)

    X = work[feat_cols].values.astype("float64")
    X_scaled = artifact["scaler"].transform(X)
    work["if_pred"] = artifact["model"].predict(X_scaled)
    work["if_score"] = artifact["model"].decision_function(X_scaled)
    work["if_alert"] = (work["if_pred"] == -1).astype(int)

    p95 = artifact.get("p95_thresholds")
    if not p95:
        raise ValueError("joblib 无 p95_thresholds，请用四维流水线重新训练后推理。")
    work["rule_alert"] = _apply_p95_rule_alerts(work, p95, rt_key)
    work["combined_alert"] = ((work["if_alert"] == 1) | (work["rule_alert"] == 1)).astype(int)
    work["p95_cpu_thr"] = p95["cpu"]
    work["p95_rt_thr"] = p95["rt"]
    return work


def save_infer_alerts(df: pd.DataFrame, output_path: str | os.PathLike[str]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return str(output_path)


def summarize_alerts(df: pd.DataFrame) -> dict[str, int | float]:
    n = len(df)
    return {
        "rows": n,
        "if_anomaly": int((df["if_alert"] == 1).sum()),
        "rule_alert": int((df["rule_alert"] == 1).sum()),
        "combined_alert": int((df["combined_alert"] == 1).sum()),
        "if_anomaly_ratio": float((df["if_alert"] == 1).mean()) if n else 0.0,
    }
