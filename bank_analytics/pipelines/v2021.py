"""
Alibaba v2021 流水线编排：数据处理 → 异常检测(IF+P95) → 短窗容量(QPS驱动)。
"""

from __future__ import annotations

import os

from bank_analytics.settings import Settings
from bank_analytics.v2021_capacity import run_short_window_capacity
from bank_analytics.v2021_data import prepare_v2021_data
from bank_analytics.v2021_model import (
    run_isolation_forest,
    save_model_artifact,
    save_result_plots,
)


def run_v2021(settings: Settings) -> None:
    out = settings.output_dir_v2021
    os.makedirs(out, exist_ok=True)

    prepared = prepare_v2021_data(settings, out)

    result = run_isolation_forest(
        prepared.instance_df,
        prepared.feat_spec,
        msinstanceid=prepared.msinstanceid,
        contamination=settings.if_contamination,
    )
    save_model_artifact(result, out)
    save_result_plots(result.scored_df, out)

    if settings.capacity_enabled:
        run_short_window_capacity(
            prepared.instance_df,
            result.split_idx,
            out,
            cpu_threshold=settings.capacity_cpu_threshold,
            mem_threshold=settings.capacity_memory_threshold,
        )

    print("[INFO] v2021 流水线完成，输出目录:", out)

    if not settings.try_prophet_v2021:
        return
    _optional_prophet_demo(result.scored_df, result.split_idx, out)


def _optional_prophet_demo(d, split_idx: int, output_dir: str | os.PathLike[str]) -> None:
    try:
        from prophet import Prophet
    except ImportError:
        print("[WARN] 未安装 prophet，跳过。安装: pip install prophet")
        return

    pt = d[["ds", "instance_cpu_usage", "throughput_total"]].rename(
        columns={"instance_cpu_usage": "y"}
    )
    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
    m.add_regressor("throughput_total")
    m.fit(pt.iloc[:split_idx])
    future = m.predict(pt.iloc[split_idx:])
    fig3 = m.plot(future)
    fig3.savefig(os.path.join(output_dir, "fig_v2021_prophet_cpu_optional.png"), dpi=150)
    print("[INFO] 可选 Prophet 图已保存 fig_v2021_prophet_cpu_optional.png")
