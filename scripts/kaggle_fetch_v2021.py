"""
Kaggle Notebook: download Alibaba v2021 shards via OSS wget.

Tar/extract under /kaggle/tmp (scratch); only move *.csv into DATA to save
/kaggle/working quota (~20GB).

Usage in a notebook cell:
    %run /kaggle/working/bank-analytics/scripts/kaggle_fetch_v2021.py

Or paste the body of fetch_all_shards() into a cell.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

OSS = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces"
DATA = Path("/kaggle/working/data/v2021")
TMP = Path("/kaggle/tmp/v2021_fetch")

MSRT_SHARDS = list(range(8))
RES_SHARDS = list(range(4))

MIN_CSV_BYTES = 50_000_000  # ~50MB; smaller files treated as corrupt


def _run(cmd: list[str], *, label: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(f"{label} failed (exit {r.returncode}): {err[:500]}")


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 2**30


def fetch_shard(kind: str, i: int) -> Path:
    name = f"{kind}_{i}"
    csv = DATA / f"{name}.csv"
    if csv.is_file() and csv.stat().st_size >= MIN_CSV_BYTES:
        print(f"[skip] {name}.csv ({csv.stat().st_size / 2**20:.1f} MB)")
        return csv
    if csv.is_file():
        print(f"[del partial] {csv} ({csv.stat().st_size} bytes)")
        csv.unlink()

    work = TMP / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    tar = work / f"{name}.tar.gz"
    url = f"{OSS}/{kind}/{name}.tar.gz"
    print(f"[wget] {name}  (working free {_disk_free_gb(DATA):.1f} GiB)")
    _run(
        ["wget", "--tries=5", "--timeout=600", "-O", str(tar), url],
        label=f"wget {name}",
    )
    tar_mb = tar.stat().st_size / 2**20
    print(f"       tar {tar_mb:.1f} MB")
    if tar_mb < 10:
        tar.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: tar too small ({tar_mb:.1f} MB)")

    _run(["gzip", "-t", str(tar)], label=f"gzip -t {name}")
    _run(["tar", "-xzf", str(tar), "-C", str(work)], label=f"tar {name}")

    hits = list(work.rglob(f"{name}.csv"))
    if not hits:
        raise FileNotFoundError(f"{name}.csv not found under {work}")
    shutil.move(str(hits[0]), str(csv))
    shutil.rmtree(work, ignore_errors=True)

    if csv.stat().st_size < MIN_CSV_BYTES:
        csv.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: csv too small after extract")

    print(f"[ok]   {csv.name} ({csv.stat().st_size / 2**20:.1f} MB)")
    return csv


def cleanup_data_tars() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for t in DATA.glob("*.tar.gz"):
        t.unlink()
        print(f"[cleanup] removed {t.name} from DATA")


def fetch_all_shards(
    msrt_shards: list[int] | None = None,
    res_shards: list[int] | None = None,
) -> None:
    msrt_shards = msrt_shards if msrt_shards is not None else MSRT_SHARDS
    res_shards = res_shards if res_shards is not None else RES_SHARDS
    DATA.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    cleanup_data_tars()

    for i in msrt_shards:
        fetch_shard("MSRTQps", i)
    for i in res_shards:
        fetch_shard("MSResource", i)

    print("\n=== done ===")
    for p in sorted(DATA.glob("*.csv")):
        print(f"  {p.name}  {p.stat().st_size / 2**20:.1f} MB")
    print(f"working free: {_disk_free_gb(DATA):.1f} GiB")


if __name__ == "__main__":
    fetch_all_shards()
