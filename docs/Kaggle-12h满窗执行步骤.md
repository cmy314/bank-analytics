# Kaggle 12 小时满窗 — 执行步骤（静默下载版）

> **重要**：下载必须用本文 **`import scripts.kaggle_fetch_v2021`**，不要用 `!wget` 或旧版内联代码，否则仍会刷进度条。  
> 代码版本需 **≥ 0.1.2**（含 `v2021-merge-train`、后半段可无 0 号分片）。

---

## 0. 固定 anchor（你的实验）

```text
MSNAME_FILTER=315f624d255fc7680cb985a3e0796397
MSINSTANCEID_FILTER=00008c44230137c49ee03f7b952b7241
```

（若 `shard_anchor.json` 里是更长完整字符串，以文件为准。）

---

## Notebook 公共单元格（每个 Notebook 先跑）

### 单元格 1 — 克隆 + 依赖

```python
import os, sys
from pathlib import Path

os.chdir("/kaggle/working")
!rm -rf /kaggle/working/bank-analytics
!git clone -q https://github.com/cmy314/bank-analytics.git /kaggle/working/bank-analytics

CODE = "/kaggle/working/bank-analytics"
sys.path.insert(0, CODE)
os.chdir(CODE)

!pip install -q pandas numpy scikit-learn matplotlib joblib pyarrow python-dotenv
!python -m bank_analytics --version
```

应看到 `0.1.2` 或更高。

### 单元格 2 — 确认下载脚本为静默版

```python
from pathlib import Path
src = Path("/kaggle/working/bank-analytics/scripts/kaggle_fetch_v2021.py").read_text()
assert "DEVNULL" in src or "--no-verbose" in src, "GitHub 代码过旧，请本地 push 后重 clone"
print("download script: quiet OK")
```

### 单元格 3 — 静默下载（**只打 [skip]/[ok]**）

```python
import os, sys
sys.path.insert(0, "/kaggle/working/bank-analytics")

os.environ["DATA_DIR_V2021"] = "/kaggle/tmp/v2021_data"

from scripts.kaggle_fetch_v2021 import fetch_all_shards

# ↓↓↓ 按 Notebook 改分片列表（见下文 Part A / Part B）↓↓↓
fetch_all_shards(
    msrt_shards=list(range(12)),   # Part A: 0-11；Part B 改为 range(12,25)
    res_shards=list(range(6)),     # Part A: 0-5；  Part B 改为 range(6,12)
)
```

输出示例：

```text
[skip] MSRTQps_0
[ok] MSRTQps_11
[done] 18 csv in /kaggle/tmp/v2021_data
```

**禁止**使用：

```python
!wget ...          # 会刷进度
%run ...           # 若脚本过旧也会刷进度；优先用上面 import 方式
```

---

## Notebook 1 — Part A（MSRT 0–11，Resource 0–5，约 6h）

### 单元格 4 — 下载

```python
fetch_all_shards(msrt_shards=list(range(12)), res_shards=list(range(6)))
```

### 单元格 5 — 跑流水线

```python
import os
os.chdir("/kaggle/working/bank-analytics")

os.environ["DATA_DIR_V2021"] = "/kaggle/tmp/v2021_data"
os.environ["OUTPUT_DIR_V2021"] = "/kaggle/working/output/part_a"
os.environ["MERGE_STRATEGY"] = "asof"
os.environ["MERGE_ASOF_TOLERANCE_MS"] = "45000"
os.environ["DISK_ACCUMULATE"] = "true"
os.environ["ONLY_FIRST_MSNAME"] = "false"
os.environ["MSNAME_FILTER"] = "315f624d255fc7680cb985a3e0796397"
os.environ["MSINSTANCEID_FILTER"] = "00008c44230137c49ee03f7b952b7241"

!python -m bank_analytics v2021-shards \
  --msrt-shards 0,1,2,3,4,5,6,7,8,9,10,11 \
  --resource-shards 0,1,2,3,4,5
```

### 单元格 6 — 打包 Part A

```python
import shutil
from pathlib import Path
staging = Path("/kaggle/working/part_a_zip")
staging.mkdir(exist_ok=True)
for f in ["merged_v2021_multishard.parquet", "shard_anchor.json"]:
    shutil.copy2(f"/kaggle/working/output/part_a/{f}", staging / f)
shutil.make_archive("/kaggle/working/part_a", "zip", staging)
print("下载 part_a.zip")
```

---

## Notebook 2 — Part B（MSRT 12–24，Resource 6–11，约 6h）

CSV 仍在 `/kaggle/tmp/v2021_data`（若会话过期需重下 Part B 分片）。

### 单元格 4 — 只下后半分片

```python
fetch_all_shards(msrt_shards=list(range(12, 25)), res_shards=list(range(6, 12)))
```

### 单元格 5 — 跑流水线（**无 0 号片**，必须 preset anchor）

```python
import os
os.chdir("/kaggle/working/bank-analytics")

os.environ["DATA_DIR_V2021"] = "/kaggle/tmp/v2021_data"
os.environ["OUTPUT_DIR_V2021"] = "/kaggle/working/output/part_b"
os.environ["MERGE_STRATEGY"] = "asof"
os.environ["DISK_ACCUMULATE"] = "true"
os.environ["ONLY_FIRST_MSNAME"] = "false"
os.environ["MSNAME_FILTER"] = "315f624d255fc7680cb985a3e0796397"
os.environ["MSINSTANCEID_FILTER"] = "00008c44230137c49ee03f7b952b7241"

!python -m bank_analytics v2021-shards \
  --msrt-shards 12,13,14,15,16,17,18,19,20,21,22,23,24 \
  --resource-shards 6,7,8,9,10,11
```

### 单元格 6 — 打包

```python
import shutil
from pathlib import Path
staging = Path("/kaggle/working/part_b_zip")
staging.mkdir(exist_ok=True)
shutil.copy2("/kaggle/working/output/part_b/merged_v2021_multishard.parquet", staging)
shutil.make_archive("/kaggle/working/part_b", "zip", staging)
print("下载 part_b.zip")
```

---

## Notebook 3 — 合并 12h + 训练

上传 `part_a.zip`、`part_b.zip` 为 Kaggle Dataset，或放到 `/kaggle/input/`。

```python
import os, sys, zipfile
from pathlib import Path

CODE = "/kaggle/working/bank-analytics"
sys.path.insert(0, CODE)
os.chdir(CODE)

# 解压（路径按你的 Dataset 改）
# !unzip -q /kaggle/input/xxx/part_a.zip -d /kaggle/working/merge/part_a
# !unzip -q /kaggle/input/xxx/part_b.zip -d /kaggle/working/merge/part_b

os.environ["OUTPUT_DIR_V2021"] = "/kaggle/working/output/v2021_12h"
os.environ["MSNAME_FILTER"] = "315f624d255fc7680cb985a3e0796397"
os.environ["MSINSTANCEID_FILTER"] = "00008c44230137c49ee03f7b952b7241"
os.environ["ONLY_FIRST_MSNAME"] = "false"

!python -m bank_analytics v2021-merge-train \
  --parts /kaggle/working/merge/part_a,/kaggle/working/merge/part_b
```

### 验收

```python
import json, pandas as pd
from pathlib import Path
OUT = Path("/kaggle/working/output/v2021_12h")
cap = pd.read_csv(OUT / "capacity_forecast_v2021.csv")
print("点数:", len(cap))
if "ds" in cap.columns:
    print("跨度:", pd.to_datetime(cap["ds"]).max() - pd.to_datetime(cap["ds"]).min())
```

目标：**点数 ≥ 600，跨度 ≥ 11 小时**。

### 打包最终结果

```python
import shutil
shutil.make_archive("/kaggle/working/kaggle_results_12h", "zip",
                    "/kaggle/working/output/v2021_12h")
```

---

## 仍看到 wget 进度？

| 原因 | 处理 |
|------|------|
| 用了 `!wget` | 删掉，改用 `fetch_all_shards` |
| GitHub 未 push 最新脚本 | 本地 `git push` 后重 `git clone` |
| 用了对话里旧的内联 `fetch()` | 只用本文 **单元格 3** |
| 验证 | 跑单元格 2，`quiet OK` 必须通过 |

---

## 本机 push（在跑 Kaggle 前）

```powershell
cd F:\bank-analytics
git add scripts/kaggle_fetch_v2021.py
git commit -m "fix: silent shard download (DEVNULL)"
git push
```
