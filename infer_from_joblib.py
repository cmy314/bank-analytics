#!/usr/bin/env python3
"""
压测后一条命令：Prometheus 导出 CSV + if_v2021.joblib → 告警表。

示例：
  cd analytics
  python infer_from_joblib.py --csv output/prometheus_txn.csv --model output/v2021/if_v2021.joblib
  python -m bank_analytics v2021-infer --csv ... --model ...
"""

from __future__ import annotations

import argparse
import sys

from bank_analytics.infer_joblib import infer_alerts_from_csv, save_infer_alerts, summarize_alerts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prometheus CSV + if_v2021.joblib → IF/P95 告警表",
    )
    parser.add_argument("--csv", required=True, help="Grafana/Prometheus 导出的宽表 CSV")
    parser.add_argument("--model", required=True, help="训练产出的 if_v2021.joblib")
    parser.add_argument(
        "--out",
        default="output/infer_alerts.csv",
        help="告警结果 CSV（默认 output/infer_alerts.csv）",
    )
    parser.add_argument("--rt-unit", choices=["auto", "ms", "s"], default="auto")
    parser.add_argument("--cpu-unit", choices=["auto", "percent", "ratio"], default="auto")
    parser.add_argument("--memory-unit", choices=["auto", "percent", "ratio"], default="auto")
    args = parser.parse_args(argv)

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
    if stats["combined_alert"]:
        bad = df.loc[df["combined_alert"] == 1, ["ds", "throughput_total", "rt", "if_score"]]
        print("[INFO] 合并告警样例（前 5 行）:")
        print(bad.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
