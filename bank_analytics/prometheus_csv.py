"""
Prometheus / Grafana 导出 CSV → 与训练一致的四维宽表列名。
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

STANDARD_COLS = (
    "timestamp",
    "throughput_total",
    "rt",
    "instance_cpu_usage",
    "instance_memory_usage",
)

COLUMN_ALIASES: dict[str, list[str]] = {
    "timestamp": [
        "timestamp",
        "time",
        "Time",
        "unix",
        "ts",
        "datetime",
        "Date",
    ],
    "throughput_total": [
        "throughput_total",
        "qps",
        "http_qps",
        "bank:http_qps:rate1m",
        "bank_http_qps_rate1m",
        "HTTP_MCR",
        "http_mcr",
        "Value #A",
        "Value",
    ],
    "rt": [
        "rt",
        "HTTP_RT",
        "http_rt",
        "latency",
        "latency_p95",
        "bank:http_latency_p95:5m",
        "bank_http_latency_p95_5m",
        "p95_rt",
        "http_latency_p95",
        "Value #B",
    ],
    "instance_cpu_usage": [
        "instance_cpu_usage",
        "cpu",
        "cpu_usage",
        "process_cpu_usage",
        "bank_cpu",
        "Value #C",
    ],
    "instance_memory_usage": [
        "instance_memory_usage",
        "memory",
        "memory_usage",
        "jvm_memory_ratio",
        "heap_ratio",
        "bank_memory",
        "Value #D",
    ],
}


def _normalize_header(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lstrip("\ufeff"))


def _pick_column(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    cols = {_normalize_header(c): c for c in df.columns}
    for alias in aliases:
        key = _normalize_header(alias)
        if key in cols:
            return cols[key]
    for alias in aliases:
        key = _normalize_header(alias).lower()
        for norm, orig in cols.items():
            if norm.lower() == key:
                return orig
    return None


def _to_epoch_ms(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        s = pd.to_numeric(series, errors="coerce")
        if s.dropna().empty:
            return s
        median = float(s.dropna().median())
        if median > 1e12:
            return s.astype("int64")
        if median > 1e9:
            return (s * 1000).astype("int64")
        return s.astype("int64")
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return (parsed.view("int64") // 1_000_000).astype("Int64")


def _scale_ratio_to_percent(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    if float(s.max()) <= 1.5:
        return s * 100.0
    return s


def _scale_seconds_to_ms(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    if float(s.max()) <= 30.0:
        return s * 1000.0
    return s


def normalize_prometheus_csv(
    df: pd.DataFrame,
    *,
    rt_unit: str = "auto",
    cpu_unit: str = "auto",
    memory_unit: str = "auto",
) -> pd.DataFrame:
    """将 Grafana/Prometheus 宽表映射为训练用四维列。"""
    raw = df.copy()
    raw.columns = [_normalize_header(c) for c in raw.columns]

    mapping: dict[str, str] = {}
    for std, aliases in COLUMN_ALIASES.items():
        picked = _pick_column(raw, aliases)
        if picked is not None:
            mapping[std] = picked

    missing = [c for c in STANDARD_COLS if c not in mapping]
    if missing:
        raise ValueError(
            f"CSV 缺少可映射列: {missing}。当前列: {raw.columns.tolist()}。"
            "请从 Grafana 导出 QPS/P95/CPU/Memory 四列，或参考 docs/Prometheus-CSV推理.md。"
        )

    out = pd.DataFrame()
    out["timestamp"] = _to_epoch_ms(raw[mapping["timestamp"]])
    out["throughput_total"] = pd.to_numeric(raw[mapping["throughput_total"]], errors="coerce")
    out["rt"] = pd.to_numeric(raw[mapping["rt"]], errors="coerce")
    out["instance_cpu_usage"] = pd.to_numeric(raw[mapping["instance_cpu_usage"]], errors="coerce")
    out["instance_memory_usage"] = pd.to_numeric(
        raw[mapping["instance_memory_usage"]], errors="coerce"
    )

    if rt_unit == "auto":
        out["rt"] = _scale_seconds_to_ms(out["rt"])
    elif rt_unit == "s":
        out["rt"] = out["rt"] * 1000.0

    if cpu_unit == "auto":
        out["instance_cpu_usage"] = _scale_ratio_to_percent(out["instance_cpu_usage"])
    elif cpu_unit == "ratio":
        out["instance_cpu_usage"] = out["instance_cpu_usage"] * 100.0

    if memory_unit == "auto":
        out["instance_memory_usage"] = _scale_ratio_to_percent(out["instance_memory_usage"])
    elif memory_unit == "ratio":
        out["instance_memory_usage"] = out["instance_memory_usage"] * 100.0

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out["ds"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    return out
