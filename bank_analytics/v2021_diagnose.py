"""分片与 msname 覆盖诊断（不跑 merge / 模型）。"""

from __future__ import annotations

import os

from bank_analytics.settings import Settings, load_settings
from bank_analytics.v2021_shards import (
    diagnose_shard_coverage,
    parse_shard_list,
    resolve_msrt_and_resource_shards,
    ShardAnchor,
)


def run_v2021_diagnose(
    settings: Settings,
    msrt_shards: list[int],
    resource_shards: list[int],
) -> None:
    anchor_path = settings.output_dir_v2021 / "shard_anchor.json"
    if anchor_path.is_file():
        anchor = ShardAnchor.from_file(anchor_path)
        msname = anchor.msname
        print(f"[INFO] 使用已有 anchor: {anchor_path}")
    elif settings.msname_filter:
        msname = settings.msname_filter
        print(f"[INFO] 使用 MSNAME_FILTER")
    else:
        print("[INFO] 未找到 anchor，请先运行 v2021-shards 或设置 MSNAME_FILTER")
        msname = ""

    if not msname:
        raise SystemExit("无法诊断：需要 shard_anchor.json 或 MSNAME_FILTER")

    diagnose_shard_coverage(settings, msname, msrt_shards, resource_shards)


def main_from_env() -> None:
    settings = load_settings()
    fb = os.getenv("V2021_SHARDS", "0,1,2,3")
    msrt, res = resolve_msrt_and_resource_shards(
        os.getenv("V2021_MSRT_SHARDS"),
        os.getenv("V2021_MS_RESOURCE_SHARDS"),
        fb,
    )
    run_v2021_diagnose(settings, msrt, res)
