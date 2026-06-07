# Kaggle 多分片长序列 — 完整实现步骤

**适合：本机磁盘不够、不想上传几十 GB CSV 到 Kaggle Dataset。**  
在 Notebook 里用 `wget` 从阿里官方 OSS **按片下载** → 跑 `v2021-shards` → 只把 **产物** 拉回本地。

---

## 一、准备（5 分钟）

| 项 | 设置 |
|----|------|
| 账号 | [kaggle.com](https://www.kaggle.com) 注册并验证手机 |
| Notebook | **Code → New Notebook** |
| Accelerator | **None**（不必 GPU） |
| Internet | **On**（必须，用于 wget / git） |
| 本机 | **无需** 存放 CSV，只需能打开 Kaggle 网页 |

---

## 二、获取 `analytics` 代码（二选一）

### 方式 A：Git 公开仓库（仓库在 GitHub 且可访问时）

在 Notebook 第一个代码单元：

```python
!git clone -q https://github.com/你的用户名/bank-observability-demo.git /kaggle/working/repo
CODE = "/kaggle/working/repo/analytics"
```

### 方式 B：上传 zip 为 Kaggle Dataset（私有仓库推荐）

1. 本地把 `bank-observability-demo/analytics/` 打成 `analytics.zip`（含 `bank_analytics/` 包）。  
2. **Datasets → New Dataset** 上传，slug 如 `yourname/bank-analytics-code`。  
3. Notebook 右侧 **Add Data** 添加该 Dataset。  
4. 代码里：

```python
CODE = "/kaggle/input/bank-analytics-code/analytics"  # 按 Input 里实际路径改
import os
assert os.path.isdir(CODE), f"路径不对: {CODE}, 目录内容: {os.listdir('/kaggle/input')}"
```

---

## 三、按片下载（tar 在临时盘，只把 CSV 放进 working）

`/kaggle/working` 仅约 **20GB**。tar 与解压必须在 **`/kaggle/tmp`**，只把 `*.csv` 移到 `DATA`。

### 方式 A — 运行仓库脚本（已 clone `bank-analytics` 时）

```python
%run /kaggle/working/bank-analytics/scripts/kaggle_fetch_v2021.py
```

### 方式 B — 整段粘贴到一个单元格

```python
import shutil, subprocess
from pathlib import Path

OSS = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces"
DATA = Path("/kaggle/working/data/v2021")
TMP  = Path("/kaggle/tmp/v2021_fetch")
MSRT_SHARDS = list(range(8))
RES_SHARDS  = list(range(4))
MIN_CSV = 50_000_000

def run(cmd, label):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{label}: {(r.stderr or r.stdout)[:400]}")

def fetch(kind, i):
    name = f"{kind}_{i}"
    csv = DATA / f"{name}.csv"
    if csv.is_file() and csv.stat().st_size >= MIN_CSV:
        print(f"[skip] {name}"); return
    if csv.is_file():
        csv.unlink()
    work = TMP / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    tar = work / f"{name}.tar.gz"
    url = f"{OSS}/{kind}/{name}.tar.gz"
    run(["wget","--quiet","--tries=5","--timeout=600","-O",str(tar),url], "wget")
    run(["gzip","-t",str(tar)], "gzip")
    run(["tar","-xzf",str(tar),"-C",str(work)], "tar")
    hits = list(work.rglob(f"{name}.csv"))
    if not hits: raise FileNotFoundError(name)
    shutil.move(str(hits[0]), str(csv))
    shutil.rmtree(work, ignore_errors=True)
    print(f"[ok] {name}")

DATA.mkdir(parents=True, exist_ok=True)
for t in DATA.glob("*.tar.gz"):
    t.unlink()
for i in MSRT_SHARDS:
    fetch("MSRTQps", i)
for i in RES_SHARDS:
    fetch("MSResource", i)
print("done:", sorted(p.name for p in DATA.glob("*.csv")))
```

**检查是否下全**：

```python
DATA = Path("/kaggle/working/data/v2021")
for kind, shards in [("MSRTQps", MSRT_SHARDS), ("MSResource", RES_SHARDS)]:
    for i in shards:
        p = DATA / f"{kind}_{i}.csv"
        tag = "OK" if p.is_file() and p.stat().st_size >= MIN_CSV else "MISS"
        print(tag, p.name, f"{p.stat().st_size/2**20:.1f} MB" if p.is_file() else "")
```

> **更长窗口**：`MSRT_SHARDS = list(range(25))`、`RES_SHARDS = list(range(12))`（约 12h，易超 12 小时会话，建议分两个 Notebook 跑 0–11 与 12–24）。

---

## 四、安装依赖并跑流水线

```python
import os
import sys

sys.path.insert(0, CODE)
os.chdir(CODE)

!pip install -q pandas numpy scikit-learn matplotlib joblib pyarrow python-dotenv

OUT = "/kaggle/working/output/v2021_kaggle"
os.makedirs(OUT, exist_ok=True)

os.environ["DATA_DIR_V2021"] = str(DATA)
os.environ["OUTPUT_DIR_V2021"] = OUT
os.environ["MERGE_STRATEGY"] = "asof"
os.environ["MERGE_ASOF_TOLERANCE_MS"] = "45000"
os.environ["DISK_ACCUMULATE"] = "true"
os.environ["ONLY_FIRST_MSNAME"] = "true"
os.environ["CAPACITY_ENABLED"] = "true"
# 长序列实验不要设置 MSRT_NROWS / MS_RESOURCE_NROWS

msrt_spec = ",".join(map(str, MSRT_SHARDS))
res_spec = ",".join(map(str, RES_SHARDS))

!python -m bank_analytics v2021-shards --msrt-shards {msrt_spec} --resource-shards {res_spec}
```

### 看日志是否成功

应出现类似：

```text
[INFO] 代表实例序列: NNN 行, trace 时间戳 [...] ms, 约 MMM 分钟
```

- **NNN ≥ 100** 较理想（asof 后单实例点数）  
- **MMM ≥ 90** 表示时间窗已拉长到 1.5h 以上  

---

## 五、（建议）第二轮：固定 msname 再跑一遍

首轮会在 `OUT` 生成 `shard_anchor.json`。若后面分片曾为空，用 anchor 固定后再跑：

```python
import json
from pathlib import Path

anchor = json.loads(Path(OUT, "shard_anchor.json").read_text(encoding="utf-8"))
os.environ["MSNAME_FILTER"] = anchor["msname"]
os.environ["MSINSTANCEID_FILTER"] = anchor["msinstanceid"]
os.environ["ONLY_FIRST_MSNAME"] = "false"
print("msname / instance 已固定")

!python -m bank_analytics v2021-shards --msrt-shards {msrt_spec} --resource-shards {res_spec}
```

也可先只诊断各片是否有该 msname：

```python
!python -m bank_analytics v2021-diagnose
```

（需已设置 `MSNAME_FILTER` 或已有 `shard_anchor.json` 于 `OUTPUT_DIR_V2021`。）

---

## 六、下载结果到本机（体积很小）

```python
import shutil
from pathlib import Path

bundle = Path("/kaggle/working/kaggle_results.zip")
shutil.make_archive(str(bundle.with_suffix("")), "zip", OUT)
print("打包完成:", bundle, "大小 MB:", bundle.stat().st_size / 2**20)
```

Notebook 右侧 **Output** → 下载 `kaggle_results.zip`。

典型内容：

| 文件 | 用途 |
|------|------|
| `merged_v2021_multishard.parquet` | 拼接宽表 |
| `shard_anchor.json` | 固定服务/实例 |
| `if_v2021.joblib` | 本地/Prometheus 推理 |
| `fig_v2021_*.png`、`capacity_*` | 论文插图 |

本机 **不必** 再下载原始 CSV。

---

## 七、分片档位速查（一次 Notebook）

| 档位 | `MSRT_SHARDS` | `RES_SHARDS` | 约时长 | 说明 |
|------|---------------|--------------|--------|------|
| 试跑 | `0,1` | `0` | ~1h | 验证流程 |
| **论文推荐** | `0..7` | `0..3` | ~4h | 默认建议 |
| 加长 | `0..11` | `0..5` | ~6h | 注意 12h 超时 |
| 满窗 | `0..24` | `0..11` | ~12h | 建议 **两个 Notebook** 各跑一半 |

---

## 八、满 12h 分两次跑（高级）

**Notebook 1**（`MSRT 0–11`，`Resource 0–5`）：

```python
MSRT_SHARDS = list(range(12))
RES_SHARDS = list(range(6))
# ... fetch + run，OUT = /kaggle/working/output/part_a
```

记下 `shard_anchor.json` 里的 `MSNAME_FILTER` / `MSINSTANCEID_FILTER`。

**Notebook 2**（`MSRT 12–24`，`Resource 6–11`）：

```python
os.environ["MSNAME_FILTER"] = "..."  # 从 part_a 复制
MSRT_SHARDS = list(range(12, 25))
RES_SHARDS = list(range(6, 12))
# OUT = /kaggle/working/output/part_b
```

本机用 pandas 合并两个 `merged_v2021_multishard.parquet`（按 `timestamp, msname, msinstanceid` 去重）后再训 IF，或云端 part_b 读入 part_a 的 anchor 后只追加 wide 再 merge（需自写几行 concat，略）。

---

## 九、常见问题

| 问题 | 处理 |
|------|------|
| 检查全 MISS、目录为空 | ① 是否先跑过下载单元；② `DATA` 是否同为 `/kaggle/working/data/v2021`；③ 下载单元是否报错被忽略；④ Notebook **Internet: On** |
| `wget` 失败 | 检查 Internet: On；重跑单元；OSS 偶尔慢，加 `--tries=5` |
| 有 tar.gz 无 csv | tar 解压路径不对；改用 **§三 临时盘方案** |
| `Wrote only N of M bytes` | **working 盘满**；tar 改在 `/kaggle/tmp`，只挪 csv |
| OOM | 减少分片，如只 `MSRT 0..3`；保持 `DISK_ACCUMULATE=true` |
| 分片 1+ merge 行数 0 | 跑完 shard0 后设 `MSNAME_FILTER` 再跑；`v2021-diagnose` 看各片行数 |
| `/kaggle/working` 超 20GB | `fetch_shard` 里已 `tar.unlink`；勿在 working 里留多份 CSV 副本 |
| 会话超时 12h | 减少分片或分两个 Notebook |
| `ModuleNotFoundError` | 确认 `!pip install` 且 `sys.path.insert(0, CODE)` |

---

## 十、与本仓库命令对应

```bash
# 在 Kaggle 上等价于：
cd analytics
python -m bank_analytics v2021-shards --msrt-shards 0,1,2,3,4,5,6,7 --resource-shards 0,1,2,3
```

环境变量见 `bank_analytics/settings.py`、`analytics/.env.example`。
