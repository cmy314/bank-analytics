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
    p_merge = sub.add_parser(
        "v2021-merge-train",
        help="合并多段 merged_v2021_multishard 后训练 IF+容量（12h 分两 Notebook 用）",
    )
    p_merge.add_argument(
        "--parts",
        required=True,
        help="逗号分隔：各段输出目录或 merged_v2021_multishard.parquet 路径",
    )
    p_infer = sub.add_parser(
        "v2021-infer",
        help="Prometheus 导出 CSV + if_v2021.joblib → 离线告警表",
    )
    p_infer.add_argument("--csv", required=True)
    p_infer.add_argument("--model", required=True)
    p_infer.add_argument("--out", default="output/infer_alerts.csv")
    p_infer.add_argument("--rt-unit", choices=["auto", "ms", "s"], default="auto")
    p_infer.add_argument("--cpu-unit", choices=["auto", "percent", "ratio"], default="auto")
    p_infer.add_argument("--memory-unit", choices=["auto", "percent", "ratio"], default="auto")

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
            args.msrt_shards, args.resource_shards, args.shards, settings=settings
        )
        run_v2021_shards(settings, msrt, res)
    elif args.cmd == "v2021-diagnose":
        from bank_analytics.v2021_diagnose import main_from_env

        main_from_env()
    elif args.cmd == "v2021-merge-train":
        from bank_analytics.pipelines.v2021_merge_12h import run_v2021_merge_train

        parts = [p.strip() for p in args.parts.split(",") if p.strip()]
        run_v2021_merge_train(load_settings(), parts)
    elif args.cmd == "v2021-infer":
        from bank_analytics.infer_joblib import infer_alerts_from_csv, save_infer_alerts, summarize_alerts

        cpu_unit = "ratio" if args.cpu_unit == "ratio" else args.cpu_unit
        mem_unit = "ratio" if args.memory_unit == "ratio" else args.memory_unit
        df = infer_alerts_from_csv(
            args.csv,
            args.model,
            rt_unit=args.rt_unit,
            cpu_unit=cpu_unit,
            memory_unit=mem_unit,
        )
        out = save_infer_alerts(df, args.out)
        stats = summarize_alerts(df)
        print(f"[INFO] 推理完成 → {out}")
        print(
            f"[INFO] 行数={stats['rows']}, IF异常={stats['if_anomaly']}, "
            f"P95规则={stats['rule_alert']}, 合并告警={stats['combined_alert']}"
        )
    else:
        parser.error("unknown command")

    return 0


if __name__ == "__main__":
    sys.exit(main())
