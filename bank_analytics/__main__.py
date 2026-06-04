"""
命令行入口：python -m bank_analytics v2021

PyCharm：Run Configuration → Module name: bank_analytics，Parameters: v2021
"""

from __future__ import annotations

import argparse
import os
import sys

from bank_analytics import __version__
from bank_analytics.settings import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="银行观测演示 — 离线分析（v2021 公开数据：IF + 短窗容量）",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "v2021",
        help="Alibaba microservices-v2021：MSRT+MSResource、IF+P95、QPS驱动短窗容量",
    )
    p_shards = sub.add_parser(
        "v2021-shards",
        help="多分片(_0,_1,...)固定同一 msname/实例，拼接后 IF+容量",
    )
    p_shards.add_argument(
        "--shards",
        default=os.getenv("V2021_SHARDS", "0,1,2,3"),
        help="默认同时用于 MSRT 与 MSResource；可用 --msrt-shards / --resource-shards 分开",
    )
    p_shards.add_argument("--msrt-shards", default=os.getenv("V2021_MSRT_SHARDS"))
    p_shards.add_argument("--resource-shards", default=os.getenv("V2021_MS_RESOURCE_SHARDS"))
    sub.add_parser("v2021-diagnose", help="诊断各分片是否含 anchor msname（不训练）")

    args = parser.parse_args(argv)

    if args.cmd == "v2021":
        from bank_analytics.pipelines.v2021 import run_v2021

        run_v2021(load_settings())
    elif args.cmd == "v2021-shards":
        from bank_analytics.pipelines.v2021_shards import run_v2021_shards
        from bank_analytics.v2021_shards import parse_shard_list

        settings = load_settings()
        from bank_analytics.v2021_shards import parse_shard_list, resolve_msrt_and_resource_shards

        msrt, res = resolve_msrt_and_resource_shards(
            args.msrt_shards, args.resource_shards, args.shards
        )
        run_v2021_shards(settings, msrt, res)
    elif args.cmd == "v2021-diagnose":
        from bank_analytics.v2021_diagnose import main_from_env

        main_from_env()
    else:
        parser.error("unknown command")

    return 0


if __name__ == "__main__":
    sys.exit(main())
