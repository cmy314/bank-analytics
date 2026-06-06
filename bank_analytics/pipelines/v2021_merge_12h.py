"""合并两段（或多段）多分片 merged 宽表后重新训练 IF + 容量。"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from bank_analytics.settings import Settings
from bank_analytics.v2021_capacity import run_short_window_capacity
from bank_analytics.v2021_data import MERGE_KEYS, prepare_from_merged, sort_for_merge_asof
from bank_analytics.v2021_model import (
    run_isolation_forest,
    save_model_artifact,
    save_result_plots,
)
from bank_analytics.v2021_shards import MERGED_MULTISHARD_BASENAME, save_multishard_merged


def _load_merged_part(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    for ext in (".parquet", ".csv"):
        candidate = path.parent / f"{MERGED_MULTISHARD_BASENAME}{ext}"
        if candidate.is_file():
            return _load_merged_part(candidate)
    raise FileNotFoundError(f"未找到 merged 宽表: {path}")


def merge_merged_parts(part_paths: list[str | os.PathLike[str]]) -> pd.DataFrame:
    """按 timestamp+msname+msinstanceid 去重合并多段 merged 宽表。"""
    frames: list[pd.DataFrame] = []
    for raw in part_paths:
        p = Path(raw)
        if p.is_dir():
            df = _load_merged_part(p)
        else:
            df = _load_merged_part(p)
        print(f"[INFO] 载入 {p} shape={df.shape}")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=list(MERGE_KEYS), keep="first")
    combined = sort_for_merge_asof(combined)
    print(f"[INFO] 合并去重: {before} -> {len(combined)} 行")
    return combined


def run_v2021_merge_train(settings: Settings, part_paths: list[str | os.PathLike[str]]) -> None:
    out = settings.output_dir_v2021
    os.makedirs(out, exist_ok=True)
    merged = merge_merged_parts(part_paths)
    save_multishard_merged(merged, out)

    prepared = prepare_from_merged(merged, settings)
    inst_df = prepared.instance_df
    t_min = inst_df["timestamp"].min() if len(inst_df) else None
    t_max = inst_df["timestamp"].max() if len(inst_df) else None
    span_min = (t_max - t_min) / 60_000 if t_min is not None and t_max is not None else 0
    print(
        f"[INFO] 代表实例序列: {len(inst_df)} 行, "
        f"trace 时间戳 [{t_min} .. {t_max}] ms, 约 {span_min:.0f} 分钟"
    )

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

    print("[INFO] 12h 合并训练完成")
    print(f"[INFO]   输出: {out}")
