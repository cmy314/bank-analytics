"""
Alibaba v2021 — 模型层（与数据处理解耦）。

职责：五维 Isolation Forest 训练/打分、P95 规则、joblib 持久化、结果图。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from bank_analytics.v2021_data import FiveDimFeatureSpec


@dataclass
class IsolationForestRunResult:
    scored_df: pd.DataFrame
    model: IsolationForest
    scaler: StandardScaler
    feat_spec: FiveDimFeatureSpec
    split_idx: int
    msinstanceid: str


def _apply_p95_rule_alerts(
    df: pd.DataFrame,
    split_idx: int,
    rt_key: str,
    er_key: str,
) -> pd.Series:
    train = df.iloc[:split_idx]
    train_cpu_thr = train["instance_cpu_usage"].quantile(0.95)
    train_rt_thr = train[rt_key].quantile(0.95)
    rule = (df["instance_cpu_usage"] > train_cpu_thr) | (df[rt_key] > train_rt_thr)
    if float(df[er_key].max()) > 0.0:
        train_er_thr = train[er_key].quantile(0.95)
        rule = rule | (df[er_key] > train_er_thr)
    return rule.astype(int)


def run_isolation_forest(
    instance_df: pd.DataFrame,
    feat_spec: FiveDimFeatureSpec,
    msinstanceid: str,
    contamination: float,
    train_ratio: float = 0.8,
) -> IsolationForestRunResult:
    d = instance_df.copy()
    X = d[feat_spec.feat_cols].values.astype("float64")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split_idx = int(len(X_scaled) * train_ratio)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_scaled[:split_idx])

    d["if_pred"] = model.predict(X_scaled)
    d["if_score"] = model.decision_function(X_scaled)
    d["rule_alert"] = _apply_p95_rule_alerts(
        d, split_idx, feat_spec.rt_key, feat_spec.error_rate_key
    )

    return IsolationForestRunResult(
        scored_df=d,
        model=model,
        scaler=scaler,
        feat_spec=feat_spec,
        split_idx=split_idx,
        msinstanceid=msinstanceid,
    )


def save_model_artifact(result: IsolationForestRunResult, output_dir: str | os.PathLike[str]) -> str:
    path = os.path.join(output_dir, "if_v2021.joblib")
    joblib.dump(
        {
            "model": result.model,
            "scaler": result.scaler,
            "feat_cols": result.feat_spec.feat_cols,
            "feat_semantic": result.feat_spec.feat_semantic,
            "msinstanceid_sample": result.msinstanceid,
            "split_idx": result.split_idx,
            "note": "五维对齐: TPS/RT/ErrorRate/CPU/Memory",
        },
        path,
    )
    print(f"[INFO] 模型已保存 {path}")
    return path


def save_result_plots(scored_df: pd.DataFrame, output_dir: str | os.PathLike[str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(scored_df["ds"], scored_df["throughput_total"], label="throughput_total")
    ax.legend()
    ax.set_title("v2021 throughput_total (sample instance)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_v2021_throughput.png"), dpi=150)

    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(scored_df["ds"], scored_df["if_score"], label="IF score (lower = more anomalous)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "fig_v2021_if_score.png"), dpi=150)
