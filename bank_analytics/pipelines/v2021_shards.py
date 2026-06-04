"""多分片 v2021 编排：拼接宽表 → IF + 容量。"""

from __future__ import annotations

import os

from bank_analytics.settings import Settings
from bank_analytics.v2021_capacity import run_short_window_capacity
from bank_analytics.v2021_shards import resolve_data_dir, resolve_msrt_and_resource_shards
from bank_analytics.v2021_model import (
    run_isolation_forest,
    save_model_artifact,
    save_result_plots,
)
from bank_analytics.v2021_shards import run_multi_shard_pipeline


def run_v2021_shards(
    settings: Settings,
    msrt_shards: list[int],
    resource_shards: list[int],
) -> None:
    out = settings.output_dir_v2021
    os.makedirs(out, exist_ok=True)
    data_dir = resolve_data_dir(settings)
    print(f"[INFO] DATA_DIR_V2021 = {data_dir}")
    print(f"[INFO] MERGE_STRATEGY = {settings.merge_strategy}")
    print(f"[INFO] MSRT 分片 = {msrt_shards}")
    print(f"[INFO] MSResource 分片 = {resource_shards}")
    if settings.msrt_nrows or settings.ms_resource_nrows:
        print(
            "[WARN] 已设置 MSRT_NROWS / MS_RESOURCE_NROWS，会截断数据；"
            "长序列实验请删除这两项"
        )

    anchor, prepared = run_multi_shard_pipeline(settings, msrt_shards, resource_shards)

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

    print("[INFO] v2021 多分片流水线完成")
    print(f"[INFO]   MSRT 分片: {anchor.msrt_shards}")
    print(f"[INFO]   MSResource 分片: {anchor.ms_resource_shards}")
    print(f"[INFO]   输出: {out}")
