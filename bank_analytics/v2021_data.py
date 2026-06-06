"""
Alibaba v2021 — 数据处理层（与模型解耦）。

职责：读 CSV → 透视 → merge → 派生列 → 落盘 merged_v2021；
      选单实例 → 解析五维特征列（TPS/RT/ErrorRate/CPU/Memory）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from bank_analytics.settings import Settings

MERGE_KEYS = ["timestamp", "msname", "msinstanceid"]
ASOF_BY_KEYS = ["msname", "msinstanceid"]


def sort_for_merge_asof(df: pd.DataFrame, by_keys: list[str] | None = None) -> pd.DataFrame:
    """merge_asof 前置：统一键类型、去重，并按 by + timestamp 排序。"""
    if df.empty:
        return df.copy()
    keys = by_keys or ASOF_BY_KEYS
    out = df.copy()
    for col in keys:
        if col not in out.columns:
            raise ValueError(f"merge_asof 缺少列 {col}，实际列: {out.columns.tolist()}")
        out[col] = out[col].astype(str)
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    out = out.dropna(subset=keys + ["timestamp"])
    out = out.drop_duplicates(subset=keys + ["timestamp"], keep="first")
    out["timestamp"] = out["timestamp"].astype("int64")
    return out.sort_values(keys + ["timestamp"], kind="mergesort").reset_index(drop=True)


def merge_asof_by_instance(
    wide: pd.DataFrame,
    res: pd.DataFrame,
    *,
    on: str = "timestamp",
    by: list[str] | None = None,
    tolerance: int,
) -> pd.DataFrame:
    """
    按 (msname, msinstanceid) 分组做 merge_asof，避免全局 by= 排序在跨分片拼接后失败。
    """
    by = by or ASOF_BY_KEYS
    wide_s = sort_for_merge_asof(wide, by)
    res_s = sort_for_merge_asof(res, by)
    if wide_s.empty:
        return wide_s
    res_value_cols = [c for c in res_s.columns if c not in set(by) | {on}]
    parts: list[pd.DataFrame] = []
    n_groups = 0
    for key_vals, left_g in wide_s.groupby(by, sort=False):
        n_groups += 1
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        mask = pd.Series(True, index=res_s.index)
        for col, val in zip(by, key_vals):
            mask &= res_s[col] == str(val)
        right_g = res_s.loc[mask]
        left_g = left_g.sort_values(on, kind="mergesort").reset_index(drop=True)
        if right_g.empty:
            parts.append(left_g)
            continue
        right_g = right_g.sort_values(on, kind="mergesort").reset_index(drop=True)
        right_part = right_g[[on] + res_value_cols]
        parts.append(
            pd.merge_asof(
                left_g,
                right_part,
                on=on,
                direction="nearest",
                tolerance=tolerance,
            )
        )
    merged = pd.concat(parts, ignore_index=True)
    print(f"[INFO] merge_asof 按实例分组: {n_groups} 组, shape={merged.shape}")
    return merged


@dataclass(frozen=True)
class FiveDimFeatureSpec:
    """孤立森林输入的五列及其语义标签（供 v2021_model 使用）。"""

    feat_cols: list[str]
    feat_semantic: list[str]
    rt_key: str
    error_rate_key: str


@dataclass(frozen=True)
class V2021PreparedData:
    """数据处理阶段产出，可直接交给模型模块。"""

    merged: pd.DataFrame
    msinstanceid: str
    instance_df: pd.DataFrame
    feat_spec: FiveDimFeatureSpec


def read_msrt(path: str, nrows: int | None) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    metric_col = "metric" if "metric" in df.columns else "metrics"
    if metric_col not in df.columns:
        raise ValueError(f"找不到 metric/metrics 列，当前列: {df.columns.tolist()}")
    return df, metric_col


def read_ms_resource(path: str, nrows: int | None = None) -> pd.DataFrame:
    res = pd.read_csv(path, nrows=nrows, low_memory=False)
    res.columns = [str(c).strip().lstrip("\ufeff") for c in res.columns]
    if "Unnamed: 0" in res.columns:
        res = res.drop(columns=["Unnamed: 0"])
    rename_map = {}
    if "cpu_utilization" in res.columns and "instance_cpu_usage" not in res.columns:
        rename_map["cpu_utilization"] = "instance_cpu_usage"
    if "memory_utilization" in res.columns and "instance_memory_usage" not in res.columns:
        rename_map["memory_utilization"] = "instance_memory_usage"
    if rename_map:
        res = res.rename(columns=rename_map)
    need = {"timestamp", "msname", "msinstanceid", "instance_cpu_usage", "instance_memory_usage"}
    missing = need - set(res.columns)
    if missing:
        raise ValueError(f"MSResource 缺少列 {missing}，实际列: {res.columns.tolist()}")
    return res


def pivot_msrt(sub: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    wide = sub.pivot_table(
        index=MERGE_KEYS,
        columns=metric_col,
        values="value",
        aggfunc="mean",
    )
    wide = wide.reset_index()
    wide.columns.name = None
    return wide


def _resolve_msname_for_subset(df: pd.DataFrame, settings: Settings) -> tuple[str, pd.DataFrame]:
    """确定本批 MSRT 使用的 msname，并返回过滤后的长表。"""
    if settings.msname_filter:
        top_ms = settings.msname_filter
        sub = df[df["msname"] == top_ms].copy()
        if sub.empty:
            raise ValueError(
                f"MSNAME_FILTER 在当前 MSRT 分片中无匹配行: {top_ms[:48]}..."
            )
        print(f"[INFO] MSNAME_FILTER = {top_ms[:16]}... ，子集行数 = {len(sub)}")
        return top_ms, sub
    if settings.only_first_msname:
        top_ms = df["msname"].value_counts().index[0]
        sub = df[df["msname"] == top_ms].copy()
        print(f"[INFO] 仅使用 msname = {top_ms[:16]}... ，子集行数 = {len(sub)}")
        return top_ms, sub
    print("[INFO] 未过滤 msname，使用全部分片内微服务")
    return "", df.copy()


def _add_throughput_and_numeric(merged: pd.DataFrame) -> pd.DataFrame:
    mcr_cols = [c for c in merged.columns if str(c).endswith("_MCR")]
    merged["throughput_total"] = merged[mcr_cols].sum(axis=1, min_count=1)
    numeric_candidates = (
        ["throughput_total", "instance_cpu_usage", "instance_memory_usage"]
        + mcr_cols
        + [c for c in merged.columns if str(c).endswith("_RT")]
    )
    for c in numeric_candidates:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)
    merged["ds"] = pd.to_datetime(merged["timestamp"], unit="ms", utc=True)
    return merged


def build_wide_msrt(settings: Settings) -> tuple[pd.DataFrame, str]:
    """MSRT 透视宽表（不含资源列）。"""
    df, metric_col = read_msrt(str(settings.msrt_path), settings.msrt_nrows)
    top_ms, sub = _resolve_msname_for_subset(df, settings)
    wide = pivot_msrt(sub, metric_col)
    print(f"[INFO] MSRT wide shape = {wide.shape}")
    wide = _add_throughput_and_numeric(wide)
    return wide.sort_values(["msinstanceid", "timestamp"]).reset_index(drop=True), top_ms


def read_ms_resource_for_msname(settings: Settings, msname: str) -> pd.DataFrame:
    res = read_ms_resource(str(settings.ms_resource_path), settings.ms_resource_nrows)
    if msname:
        before = len(res)
        res = res[res["msname"] == msname].copy()
        print(f"[INFO] MSResource 按 msname 过滤: {before} -> {len(res)} 行")
    return res.sort_values(["msinstanceid", "timestamp"]).reset_index(drop=True)


def join_wide_with_resource(
    wide: pd.DataFrame,
    res: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """MSRT 宽表与资源表对齐。asof 可缓解 30s/60s 采样不一致导致的 inner 稀疏。"""
    if res.empty:
        print("[WARN] MSResource 为空，仅返回 MSRT 宽表（无 CPU/Memory）")
        return wide.copy()

    strategy = settings.merge_strategy
    if strategy == "asof":
        tol = settings.merge_asof_tolerance_ms
        merged = merge_asof_by_instance(wide, res, tolerance=tol)
        print(
            f"[INFO] merge_asof(tolerance={tol}ms) 完成 "
            f"(wide={len(wide)}, res={len(res)})"
        )
    else:
        try:
            merged = wide.merge(res, on=MERGE_KEYS, how="inner", validate="many_to_one")
        except pd.errors.MergeError:
            merged = wide.merge(res, on=MERGE_KEYS, how="inner")
        print(f"[INFO] inner merge shape = {merged.shape}")

    merged = _add_throughput_and_numeric(merged)
    return merged.sort_values(["msinstanceid", "ds"]).reset_index(drop=True)


def build_merged_v2021(settings: Settings) -> pd.DataFrame:
    wide, top_ms = build_wide_msrt(settings)
    res = read_ms_resource_for_msname(settings, top_ms)
    merged = join_wide_with_resource(wide, res, settings)
    print(f"[INFO] merged shape = {merged.shape}")
    return merged


def save_merged_v2021(merged: pd.DataFrame, output_dir: str | os.PathLike[str]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    base_out = os.path.join(output_dir, "merged_v2021")
    try:
        path = base_out + ".parquet"
        merged.to_parquet(path, index=False)
        print(f"[INFO] 已保存 {path}")
        return path
    except Exception as e:
        print(f"[WARN] Parquet 不可用 ({e})，改存 CSV")
        path = base_out + ".csv"
        merged.to_csv(path, index=False)
        return path


def _pick_primary_rt_column(df: pd.DataFrame) -> str | None:
    if "HTTP_RT" in df.columns:
        return "HTTP_RT"
    rt_candidates = sorted(c for c in df.columns if str(c).endswith("_RT"))
    return rt_candidates[0] if rt_candidates else None


def _ensure_error_rate_column(df: pd.DataFrame) -> str:
    for c in ("error_rate", "http_error_rate", "failure_rate"):
        if c in df.columns:
            return c
    df["error_rate"] = 0.0
    print(
        "[WARN] merged 数据中无 Error Rate 列，已使用 error_rate=0 占位。"
        "运行时请从 Prometheus 导出真实错误率并合并进宽表。"
    )
    return "error_rate"


def resolve_five_dim_features(instance_df: pd.DataFrame) -> FiveDimFeatureSpec:
    rt_key = _pick_primary_rt_column(instance_df)
    if rt_key is None:
        raise RuntimeError("宽表中不存在 RT 列（HTTP_RT 或 *_RT），无法满足五维设定")

    er_key = _ensure_error_rate_column(instance_df)
    for mandatory in ("throughput_total", "instance_cpu_usage", "instance_memory_usage"):
        if mandatory not in instance_df.columns:
            raise RuntimeError(f"五维所需列缺失: {mandatory}")

    feat_cols = [
        "throughput_total",
        rt_key,
        er_key,
        "instance_cpu_usage",
        "instance_memory_usage",
    ]
    feat_semantic = [
        "TPS(~sum MCR)",
        f"RT({rt_key})",
        f"ErrorRate({er_key})",
        "CPU",
        "Memory",
    ]
    print(f"[INFO] IF 输入五维 × 语义: {list(zip(feat_semantic, feat_cols))}")
    return FiveDimFeatureSpec(
        feat_cols=feat_cols,
        feat_semantic=feat_semantic,
        rt_key=rt_key,
        error_rate_key=er_key,
    )


def pick_instance_frame(
    merged: pd.DataFrame,
    settings: Settings,
) -> tuple[str, pd.DataFrame]:
    if settings.msinstanceid_filter:
        inst = settings.msinstanceid_filter
        instance_df = merged[merged["msinstanceid"] == inst].copy()
        if instance_df.empty:
            raise ValueError(
                f"MSINSTANCEID_FILTER 在 merged 中无匹配: {inst[:48]}..."
            )
    else:
        inst = merged["msinstanceid"].value_counts().index[0]
        instance_df = merged[merged["msinstanceid"] == inst].copy()
    print(f"[INFO] IF 使用实例 msinstanceid = {inst[:16]}... ，行数 = {len(instance_df)}")
    return inst, instance_df.sort_values("ds").reset_index(drop=True)


def prepare_from_merged(
    merged: pd.DataFrame,
    settings: Settings,
) -> V2021PreparedData:
    """在已有宽表上选实例并解析五维特征（多分片拼接后用）。"""
    inst, instance_df = pick_instance_frame(merged, settings)
    feat_spec = resolve_five_dim_features(instance_df)
    return V2021PreparedData(
        merged=merged,
        msinstanceid=inst,
        instance_df=instance_df,
        feat_spec=feat_spec,
    )


def prepare_v2021_data(settings: Settings, output_dir: str | os.PathLike[str]) -> V2021PreparedData:
    """完整数据处理：merge → 落盘 → 单实例 → 五维特征。"""
    merged = build_merged_v2021(settings)
    save_merged_v2021(merged, output_dir)
    prepared = prepare_from_merged(merged, settings)
    return prepared
