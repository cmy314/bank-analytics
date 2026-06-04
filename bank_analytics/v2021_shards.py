"""
多分片 v2021：固定同一 msname / msinstanceid，拼接后训练 IF + 容量。

MSRT 与 MSResource 分片粒度不同（官方约 25 vs 12 片），可分别配置：
  V2021_MSRT_SHARDS=0,1,2,3
  V2021_MS_RESOURCE_SHARDS=0,1

用法见 analytics/docs/长序列实验方案.md
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd

from bank_analytics.settings import Settings
from bank_analytics.v2021_data import (
    MERGE_KEYS,
    V2021PreparedData,
    build_wide_msrt,
    join_wide_with_resource,
    prepare_from_merged,
    read_ms_resource_for_msname,
    read_msrt,
    save_merged_v2021,
)

ANCHOR_FILENAME = "shard_anchor.json"
MERGED_MULTISHARD_BASENAME = "merged_v2021_multishard"

# 官方 v2021：全 trace 12h；MSRT 约 25 片 → ~29min/片；MSResource 约 12 片 → ~1h/片
MSRT_SHARD_MS = 1_728_000
MS_RESOURCE_SHARD_MS = 3_600_000
TRACE_TOTAL_MS = 43_200_000


@dataclass(frozen=True)
class ShardAnchor:
    msname: str
    msinstanceid: str
    msrt_shards: list[int]
    ms_resource_shards: list[int]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_file(cls, path: Path) -> ShardAnchor:
        data = json.loads(path.read_text(encoding="utf-8"))
        legacy = list(data.get("shard_indices", []))
        return cls(
            msname=data["msname"],
            msinstanceid=data["msinstanceid"],
            msrt_shards=list(data.get("msrt_shards", legacy)),
            ms_resource_shards=list(data.get("ms_resource_shards", legacy)),
        )


def _missing_shard_hint(data_dir: Path, filename: str) -> str:
    if not data_dir.is_dir():
        return f"\n  提示: DATA_DIR_V2021 目录不存在: {data_dir}"
    try:
        names = sorted(p.name for p in data_dir.glob("*.csv"))[:8]
    except OSError:
        names = []
    if names:
        return (
            f"\n  提示: 请检查 DATA_DIR_V2021（当前 {data_dir}）。"
            f" 目录内已有 CSV 示例: {', '.join(names)}"
        )
    return (
        f"\n  提示: 请将 {filename} 放入 DATA_DIR_V2021，或修正 .env 中的 DATA_DIR_V2021 / MSRT_PATH"
    )


def resolve_data_dir(settings: Settings) -> Path:
    if settings.data_dir_v2021 is not None:
        return settings.data_dir_v2021
    return settings.msrt_path.parent


def shard_msrt_path(data_dir: Path, shard: int) -> Path:
    p = data_dir / f"MSRTQps_{shard}.csv"
    if not p.is_file():
        raise FileNotFoundError(f"缺少 MSRT 分片: {p}{_missing_shard_hint(data_dir, p.name)}")
    return p


def shard_resource_path(data_dir: Path, shard: int) -> Path:
    p = data_dir / f"MSResource_{shard}.csv"
    if not p.is_file():
        raise FileNotFoundError(f"缺少 MSResource 分片: {p}{_missing_shard_hint(data_dir, p.name)}")
    return p


def settings_for_msrt_shard(base: Settings, shard: int, out_subdir: Path | None = None) -> Settings:
    data_dir = resolve_data_dir(base)
    out = out_subdir or base.output_dir_v2021
    os.makedirs(out, exist_ok=True)
    return replace(
        base,
        msrt_path=shard_msrt_path(data_dir, shard),
        ms_resource_path=shard_resource_path(data_dir, shard),
        output_dir_v2021=out,
    )


def settings_for_resource_shard(base: Settings, shard: int) -> Settings:
    data_dir = resolve_data_dir(base)
    return replace(
        base,
        ms_resource_path=shard_resource_path(data_dir, shard),
    )


MSRT_SHARD_MAX_INDEX = 24
MS_RESOURCE_SHARD_MAX_INDEX = 11


def parse_shard_list(spec: str, *, kind: str = "msrt") -> list[int]:
    """
    解析分片列表。kind=msrt → 0..24；kind=resource → 0..11。
    spec 为 max/all/full 时表示该类型全部分片（约 12h 窗）。
    """
    raw = spec.strip().lower()
    if raw in ("max", "all", "full"):
        end = MSRT_SHARD_MAX_INDEX if kind == "msrt" else MS_RESOURCE_SHARD_MAX_INDEX
        return list(range(end + 1))
    parts = [p.strip() for p in spec.replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("分片列表为空，示例: 0,1,2 或 max")
    return [int(p) for p in parts]


def resolve_msrt_and_resource_shards(
    msrt_spec: str | None,
    resource_spec: str | None,
    fallback: str,
) -> tuple[list[int], list[int]]:
    """解析 MSRT / MSResource 分片列表；未单独指定时共用 fallback。"""
    fb = parse_shard_list(fallback, kind="msrt")
    msrt = parse_shard_list(msrt_spec, kind="msrt") if msrt_spec else fb
    res = (
        parse_shard_list(resource_spec, kind="resource")
        if resource_spec
        else parse_shard_list(fallback, kind="resource")
    )
    if 0 not in msrt:
        raise ValueError("MSRT 分片列表须包含 0，用于确定 anchor msname")
    if 0 not in res:
        raise ValueError("MSResource 分片列表须包含 0")
    return msrt, res


def estimate_trace_span_ms(msrt_shards: list[int], resource_shards: list[int]) -> tuple[int, int]:
    msrt_ms = len(msrt_shards) * MSRT_SHARD_MS
    res_ms = len(resource_shards) * MS_RESOURCE_SHARD_MS
    return msrt_ms, res_ms


def _write_part_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def concat_parquet_parts(parts_dir: Path, label: str, batch: int = 4) -> pd.DataFrame:
    """从磁盘分批读取分片 parquet 再拼接，降低峰值内存。"""
    paths = sorted(parts_dir.glob("*.parquet"))
    if not paths:
        print(f"[WARN] {label} 无 parquet 分片: {parts_dir}")
        return pd.DataFrame()
    buffers: list[pd.DataFrame] = []
    merged_chunks: list[pd.DataFrame] = []
    for p in paths:
        buffers.append(pd.read_parquet(p))
        if len(buffers) >= batch:
            merged_chunks.append(concat_tables(buffers, label))
            buffers = []
    if buffers:
        merged_chunks.append(concat_tables(buffers, label))
    return concat_tables(merged_chunks, label)


def concat_tables(frames: list[pd.DataFrame], label: str) -> pd.DataFrame:
    valid = [f for f in frames if f is not None and len(f) > 0]
    if not valid:
        print(f"[WARN] {label} 拼接：无有效分片，返回空表")
        return pd.DataFrame()
    combined = pd.concat(valid, ignore_index=True)
    keys = list(MERGE_KEYS)
    before = len(combined)
    combined = combined.drop_duplicates(subset=keys, keep="first")
    if len(combined) < before:
        print(f"[INFO] {label} 去重: {before} -> {len(combined)} 行")
    return combined.sort_values("timestamp").reset_index(drop=True)


def count_msname_in_msrt_shard(settings: Settings, shard: int, msname: str) -> int:
    s = settings_for_msrt_shard(settings, shard, settings.output_dir_v2021)
    df, _ = read_msrt(str(s.msrt_path), s.msrt_nrows)
    return int((df["msname"] == msname).sum())


def diagnose_shard_coverage(
    settings: Settings,
    msname: str,
    msrt_shards: list[int],
    resource_shards: list[int],
) -> None:
    print(f"[INFO] ===== 分片覆盖诊断 msname={msname[:16]}... =====")
    for shard in msrt_shards:
        try:
            n = count_msname_in_msrt_shard(settings, shard, msname)
            print(f"  MSRTQps_{shard}: msname 行数 = {n}")
        except FileNotFoundError as e:
            print(f"  MSRTQps_{shard}: 缺失 ({e})")
    for shard in resource_shards:
        try:
            s = settings_for_resource_shard(settings, shard)
            res = read_ms_resource_for_msname(s, msname)
            print(f"  MSResource_{shard}: msname 行数 = {len(res)}")
        except FileNotFoundError as e:
            print(f"  MSResource_{shard}: 缺失 ({e})")
    msrt_ms, res_ms = estimate_trace_span_ms(msrt_shards, resource_shards)
    print(
        f"[INFO] 理论时间窗约 MSRT {msrt_ms/3600000:.1f}h + Resource {res_ms/3600000:.1f}h "
        f"(全 trace 12h)"
    )


def discover_anchor_from_wide(wide: pd.DataFrame, msrt_shards: list[int], res_shards: list[int]) -> ShardAnchor:
    msname = str(wide["msname"].iloc[0])
    inst = str(wide["msinstanceid"].value_counts().index[0])
    return ShardAnchor(
        msname=msname,
        msinstanceid=inst,
        msrt_shards=msrt_shards,
        ms_resource_shards=res_shards,
    )


def save_multishard_merged(merged: pd.DataFrame, output_dir: str | os.PathLike[str]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    base_out = os.path.join(output_dir, MERGED_MULTISHARD_BASENAME)
    try:
        path = base_out + ".parquet"
        merged.to_parquet(path, index=False)
    except Exception as e:
        print(f"[WARN] Parquet 不可用 ({e})，改存 CSV")
        path = base_out + ".csv"
        merged.to_csv(path, index=False)
    print(f"[INFO] 多分片拼接宽表已保存 {path}，shape = {merged.shape}")
    return path


def run_multi_shard_pipeline(
    settings: Settings,
    msrt_shards: list[int],
    resource_shards: list[int],
) -> tuple[ShardAnchor, V2021PreparedData]:
    if not msrt_shards or not resource_shards:
        raise ValueError("msrt_shards 与 resource_shards 均不能为空")

    out = settings.output_dir_v2021
    os.makedirs(out, exist_ok=True)

    msrt_ordered = sorted(dict.fromkeys(msrt_shards))
    res_ordered = sorted(dict.fromkeys(resource_shards))

    s0 = settings_for_msrt_shard(settings, msrt_ordered[0], out / "shards" / f"msrt_{msrt_ordered[0]}")
    if settings.msname_filter:
        s0 = replace(s0, msname_filter=settings.msname_filter, only_first_msname=False)
    elif settings.only_first_msname:
        s0 = replace(s0, only_first_msname=True)

    print("[INFO] ===== MSRT 分片 0: 确定 msname =====")
    wide0, msname = build_wide_msrt(s0)
    save_merged_v2021(wide0, s0.output_dir_v2021)

    anchor = discover_anchor_from_wide(wide0, msrt_ordered, res_ordered)
    if settings.msinstanceid_filter:
        anchor = replace(anchor, msinstanceid=settings.msinstanceid_filter)
    if settings.msname_filter:
        anchor = replace(anchor, msname=settings.msname_filter)

    diagnose_shard_coverage(settings, anchor.msname, msrt_ordered, res_ordered)

    wide_parts = out / "_wide_parts"
    res_parts = out / "_resource_parts"
    if settings.disk_accumulate:
        for d in (wide_parts, res_parts):
            if d.exists():
                for f in d.glob("*.parquet"):
                    f.unlink()
        _write_part_parquet(wide0, wide_parts / f"msrt_{msrt_ordered[0]}.parquet")
        print(f"[INFO] DISK_ACCUMULATE: wide 分片已落盘 {wide_parts}")
    else:
        wide_frames: list[pd.DataFrame] = [wide0]

    for shard in msrt_ordered:
        if shard == msrt_ordered[0]:
            continue
        print(f"[INFO] ===== MSRT 分片 {shard} =====")
        sx = settings_for_msrt_shard(settings, shard, out / "shards" / f"msrt_{shard}")
        sx = replace(
            sx,
            msname_filter=anchor.msname,
            only_first_msname=False,
        )
        try:
            w, _ = build_wide_msrt(sx)
            if w.empty:
                print(f"[WARN] MSRT 分片 {shard} 在 msname 下无数据，跳过")
            else:
                if settings.disk_accumulate:
                    _write_part_parquet(w, wide_parts / f"msrt_{shard}.parquet")
                else:
                    wide_frames.append(w)
                save_merged_v2021(w, sx.output_dir_v2021)
        except ValueError as e:
            print(f"[WARN] MSRT 分片 {shard}: {e}")

    if settings.disk_accumulate:
        wide_all = concat_parquet_parts(wide_parts, "MSRT wide")
    else:
        wide_all = concat_tables(wide_frames, "MSRT wide")

    res_frames: list[pd.DataFrame] = []
    for shard in res_ordered:
        print(f"[INFO] ===== MSResource 分片 {shard} =====")
        sr = settings_for_resource_shard(settings, shard)
        res = read_ms_resource_for_msname(sr, anchor.msname)
        if res.empty:
            print(f"[WARN] MSResource 分片 {shard} 在 msname 下无数据")
            continue
        if settings.disk_accumulate:
            _write_part_parquet(res, res_parts / f"res_{shard}.parquet")
        else:
            res_frames.append(res)

    if settings.disk_accumulate:
        res_all = concat_parquet_parts(res_parts, "MSResource")
    else:
        res_all = concat_tables(res_frames, "MSResource")

    print(f"[INFO] 拼接后 wide={len(wide_all)} 行, resource={len(res_all)} 行")
    merged = join_wide_with_resource(wide_all, res_all, settings)
    save_multishard_merged(merged, out)

    (out / ANCHOR_FILENAME).write_text(anchor.to_json(), encoding="utf-8")
    print(f"[INFO] anchor 已写入 {out / ANCHOR_FILENAME}")

    s_fit = replace(
        settings,
        msname_filter=anchor.msname,
        msinstanceid_filter=anchor.msinstanceid,
        only_first_msname=False,
    )
    prepared = prepare_from_merged(merged, s_fit)
    inst_df = prepared.instance_df
    t_min = inst_df["timestamp"].min() if len(inst_df) else None
    t_max = inst_df["timestamp"].max() if len(inst_df) else None
    span_min = (t_max - t_min) / 60_000 if t_min is not None and t_max is not None else 0
    print(
        f"[INFO] 代表实例序列: {len(inst_df)} 行, "
        f"trace 时间戳 [{t_min} .. {t_max}] ms, 约 {span_min:.0f} 分钟"
    )
    return anchor, prepared
