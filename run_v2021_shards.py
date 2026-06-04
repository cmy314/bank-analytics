"""
PyCharm：右键 Run；等价: python -m bank_analytics v2021-shards --shards 0,1,2

需 .env 中 DATA_DIR_V2021 指向含 MSRTQps_N.csv / MSResource_N.csv 的目录。
"""

from bank_analytics.__main__ import main

if __name__ == "__main__":
    import os

    msrt = os.getenv("V2021_MSRT_SHARDS", "max")
    res = os.getenv("V2021_MS_RESOURCE_SHARDS", "max")
    raise SystemExit(
        main(["v2021-shards", "--shards", "0", "--msrt-shards", msrt, "--resource-shards", res])
    )
