"""
v2021 短窗容量预测：QPS（吞吐）驱动 CPU/内存推演。

在观测窗内（通常约 2h 量级）用训练段拟合线性关系，在 holdout 段评估并与警戒线比较。
演示系统验证阶段使用同一公式，特征来自 Prometheus 导出宽表。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class CapacityModel:
    cpu_alpha: float
    cpu_beta: float
    mem_gamma: float
    mem_delta: float
    cpu_threshold: float
    mem_threshold: float


@dataclass(frozen=True)
class CapacityRunResult:
    scored_df: pd.DataFrame
    model: CapacityModel
    split_idx: int


def _fit_xy(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    reg = LinearRegression()
    reg.fit(x.reshape(-1, 1), y)
    return float(reg.coef_[0]), float(reg.intercept_)


def run_short_window_capacity(
    instance_df: pd.DataFrame,
    split_idx: int,
    output_dir: str | os.PathLike[str],
    *,
    cpu_threshold: float = 0.7,
    mem_threshold: float = 0.8,
) -> CapacityRunResult:
    """
    训练段拟合 CPU=α·QPS+β、Memory=γ·QPS+δ；在全序列上给出预测与触线标记。
    """
    d = instance_df.copy()
    qps = d["throughput_total"].values.astype("float64")
    cpu = d["instance_cpu_usage"].values.astype("float64")
    mem = d["instance_memory_usage"].values.astype("float64")

    train_q = qps[:split_idx]
    cpu_a, cpu_b = _fit_xy(train_q, cpu[:split_idx])
    mem_g, mem_d = _fit_xy(train_q, mem[:split_idx])

    cap = CapacityModel(
        cpu_alpha=cpu_a,
        cpu_beta=cpu_b,
        mem_gamma=mem_g,
        mem_delta=mem_d,
        cpu_threshold=cpu_threshold,
        mem_threshold=mem_threshold,
    )

    d["pred_cpu"] = cpu_a * qps + cpu_b
    d["pred_mem"] = mem_g * qps + mem_d
    d["capacity_cpu_alert"] = (d["pred_cpu"] > cpu_threshold).astype(int)
    d["capacity_mem_alert"] = (d["pred_mem"] > mem_threshold).astype(int)
    d["capacity_alert"] = (
        (d["capacity_cpu_alert"] == 1) | (d["capacity_mem_alert"] == 1)
    ).astype(int)

    _save_capacity_artifacts(d, cap, split_idx, output_dir)
    print(
        f"[INFO] 容量模型 CPU={cpu_a:.4f}·QPS+{cpu_b:.4f}, "
        f"Mem={mem_g:.4f}·QPS+{mem_d:.4f}; "
        f"警戒线 CPU>{cpu_threshold}, Mem>{mem_threshold}"
    )
    return CapacityRunResult(scored_df=d, model=cap, split_idx=split_idx)


def _save_capacity_artifacts(
    d: pd.DataFrame,
    cap: CapacityModel,
    split_idx: int,
    output_dir: str | os.PathLike[str],
) -> None:
    csv_path = os.path.join(output_dir, "capacity_forecast_v2021.csv")
    d.to_csv(csv_path, index=False)
    print(f"[INFO] 已保存 {csv_path}")

    coef_path = os.path.join(output_dir, "capacity_model_v2021.txt")
    with open(coef_path, "w", encoding="utf-8") as f:
        f.write(f"CPU = {cap.cpu_alpha} * QPS + {cap.cpu_beta}\n")
        f.write(f"Memory = {cap.mem_gamma} * QPS + {cap.mem_delta}\n")
        f.write(f"cpu_threshold={cap.cpu_threshold}\n")
        f.write(f"mem_threshold={cap.mem_threshold}\n")
        f.write(f"train_split_idx={split_idx}\n")

    if "ds" not in d.columns:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax0, ax1 = axes
    ax0.plot(d["ds"], d["instance_cpu_usage"], label="CPU 实际", alpha=0.7)
    ax0.plot(d["ds"], d["pred_cpu"], label="CPU 预测(QPS驱动)", alpha=0.8)
    ax0.axhline(cap.cpu_threshold, color="r", linestyle="--", label=f"警戒线 {cap.cpu_threshold}")
    ax0.axvline(d["ds"].iloc[split_idx], color="gray", linestyle=":", label="训练|holdout")
    ax0.set_ylabel("CPU")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.set_title("短窗容量预测 — CPU")

    ax1.plot(d["ds"], d["throughput_total"], label="QPS(吞吐)", color="green", alpha=0.7)
    ax1.set_ylabel("吞吐")
    ax1.set_xlabel("时间")
    ax1.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "fig_v2021_capacity.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] 已保存 {fig_path}")
