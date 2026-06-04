"""
从环境变量加载路径配置。

优先级：
  1) 进程环境变量（PyCharm Run Configuration / 服务器 export）
  2) 项目根目录下的 .env（需 pip install python-dotenv）

说明：不把敏感信息写进代码；数据 CSV 路径随机器变化。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _analytics_root() -> Path:
    """analytics 目录（含 bank_analytics 包的上一级）。"""
    return Path(__file__).resolve().parent.parent


def _load_dotenv_minimal(env_path: Path) -> None:
    """无 python-dotenv 时解析 analytics/.env（不覆盖已有环境变量）。"""
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _load_dotenv() -> None:
    env_path = _analytics_root() / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        _load_dotenv_minimal(env_path)


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _str_or_none(name: str) -> str | None:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return None
    return str(v).strip()


def _path_or_none(name: str) -> Path | None:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return None
    return Path(v)


def _int_or_none(name: str) -> int | None:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return None
    return int(v)


def _float_env(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return float(v)


@dataclass(frozen=True)
class Settings:
    """运行所需路径与开关。"""

    msrt_path: Path
    ms_resource_path: Path
    output_dir_v2021: Path
    msrt_nrows: int | None
    ms_resource_nrows: int | None
    only_first_msname: bool
    msname_filter: str | None
    msinstanceid_filter: str | None
    data_dir_v2021: Path | None
    merge_strategy: str
    merge_asof_tolerance_ms: int
    disk_accumulate: bool
    if_contamination: float
    capacity_enabled: bool
    capacity_cpu_threshold: float
    capacity_memory_threshold: float
    try_prophet_v2021: bool


def load_settings() -> Settings:
    _load_dotenv()
    root = _analytics_root()
    out_v2021 = Path(os.getenv("OUTPUT_DIR_V2021", str(root / "output" / "v2021")))

    return Settings(
        msrt_path=Path(os.getenv("MSRT_PATH", str(root / "data" / "v2021" / "MSRTQps_0.csv"))),
        ms_resource_path=Path(
            os.getenv("MS_RESOURCE_PATH", str(root / "data" / "v2021" / "MSResource_0.csv"))
        ),
        output_dir_v2021=out_v2021,
        msrt_nrows=_int_or_none("MSRT_NROWS"),
        ms_resource_nrows=_int_or_none("MS_RESOURCE_NROWS"),
        only_first_msname=_bool_env("ONLY_FIRST_MSNAME", True),
        msname_filter=_str_or_none("MSNAME_FILTER"),
        msinstanceid_filter=_str_or_none("MSINSTANCEID_FILTER"),
        data_dir_v2021=_path_or_none("DATA_DIR_V2021"),
        merge_strategy=os.getenv("MERGE_STRATEGY", "asof").strip().lower(),
        merge_asof_tolerance_ms=int(os.getenv("MERGE_ASOF_TOLERANCE_MS", "45000")),
        disk_accumulate=_bool_env("DISK_ACCUMULATE", True),
        if_contamination=float(os.getenv("IF_CONTAMINATION", "0.02")),
        capacity_enabled=_bool_env("CAPACITY_ENABLED", True),
        capacity_cpu_threshold=_float_env("CAPACITY_CPU_THRESHOLD", 0.7),
        capacity_memory_threshold=_float_env("CAPACITY_MEMORY_THRESHOLD", 0.8),
        try_prophet_v2021=_bool_env("TRY_PROPHET_V2021", False),
    )
