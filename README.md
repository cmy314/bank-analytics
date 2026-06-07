# bank-analytics — Python 离线分析项目

> **从主仓库拆出**：见 [docs/独立仓库迁移.md](docs/独立仓库迁移.md)。拆出后本目录即 **独立 Git 仓库根**（仓库名建议 `bank-analytics`），不再嵌套在 `bank-observability-demo/` 下。

本工程与 Java 微服务、Prometheus/Grafana **分仓库维护**，用于：

- 读取 **Alibaba microservices-v2021** 公开 trace CSV；
- **合并、特征工程**；
- **孤立森林（Isolation Forest）** + **P95 阈值规则**；
- **短窗容量预测**：QPS 驱动 CPU/内存线性推演 + 警戒线（默认 CPU 70%、内存 80%）。

Java 演示系统（另仓）负责运行时指标（Micrometer → Prometheus）；本仓负责 **离线训练、批量评估、出图**。验证时从 Prometheus 导出与训练一致的宽表（设计说明见主课题仓库 `docs/`，或本仓 `docs/`）。

---

## 一、目录结构

```
bank-analytics/              ← 独立仓库根（原 monorepo 下的 analytics/）
  README.md
  requirements.txt
  .env.example
  run_v2021.py              ← PyCharm 单分片
  run_v2021_shards.py       ← PyCharm 多分片拼接
  bank_analytics/
    __main__.py             ← python -m bank_analytics v2021
    settings.py
    v2021_data.py           ← 读/merge/特征
    v2021_model.py          ← IF + P95 + joblib
    v2021_capacity.py       ← QPS 驱动短窗容量
    pipelines/v2021.py      ← 编排
  data/v2021/               ← MSRTQps_*.csv、MSResource_*.csv（勿提交 Git）
  output/v2021/             ← 运行生成
```

---

## 二、环境准备

```powershell
cd F:\bank-observability-demo\analytics
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，填写 `MSRT_PATH`、`MS_RESOURCE_PATH`。

---

## 三、运行

```bash
cd analytics
python -m bank_analytics v2021
```

或右键 `run_v2021.py` → Run。

**多分片（固定同一微服务/实例，拉长时间序列）**：

```bash
# .env 设置 DATA_DIR_V2021、V2021_SHARDS=0,1,2
python -m bank_analytics v2021-shards --shards 0,1,2
```

详见 **[docs/多分片固定服务实验.md](docs/多分片固定服务实验.md)**。

---

## 四、运行产物

| 文件 | 含义 |
|------|------|
| `merged_v2021.parquet` / `.csv` | 合并宽表 |
| `if_v2021.joblib` | Scaler + 孤立森林（**四维**：QPS/RT/CPU/Memory） |
| `fig_v2021_*.png` | 吞吐、IF 得分 |
| `capacity_forecast_v2021.csv` | 容量预测与触线标记 |
| `capacity_model_v2021.txt` | α、β、γ、δ 系数 |
| `fig_v2021_capacity.png` | CPU 实际 vs QPS 驱动预测 |
| `fig_v2021_prophet_cpu_optional.png` | 仅当 `TRY_PROPHET_V2021=true` |
| `infer_alerts.csv` | 压测后 Prometheus CSV 推理告警表 |

## 五、压测后离线推理（Prometheus CSV）

```bash
python infer_from_joblib.py \
  --csv output/prometheus_txn.csv \
  --model output/v2021/if_v2021.joblib \
  --out output/infer_alerts.csv
```

详见 **[docs/Prometheus-CSV推理.md](docs/Prometheus-CSV推理.md)**。旧五维 joblib 需重新 `v2021-merge-train`。

---

## 六、环境变量

| 变量 | 含义 |
|------|------|
| `MSRT_PATH` / `MS_RESOURCE_PATH` | v2021 CSV |
| `DATA_DIR_V2021` | 多分片 CSV 目录 |
| `V2021_SHARDS` | 默认 `0,1`；命令行 `--shards` 可覆盖 |
| `MSNAME_FILTER` / `MSINSTANCEID_FILTER` | 跨分片固定服务/实例 |
| `CAPACITY_ENABLED` | 是否跑短窗容量（默认 true） |
| `CAPACITY_CPU_THRESHOLD` | CPU 警戒线（默认 0.7） |
| `CAPACITY_MEMORY_THRESHOLD` | 内存警戒线（默认 0.8） |
| `IF_CONTAMINATION` | 孤立森林 contamination |
| `TRY_PROPHET_V2021` | 可选 Prophet 演示（非主线） |

---

## 六、本地内存不够（Kaggle / Colab / Drive）

详见 **[docs/云端运行v2021指南.md](docs/云端运行v2021指南.md)**。  
**Kaggle 多分片（不占本机盘）**：[docs/Kaggle多分片运行指南.md](docs/Kaggle多分片运行指南.md)。

---

## 七、与课题范围

- **公开数据**：仅 **v2021**（已移除 v2018 集群支线）。
- **容量**：**短观测窗**（与 v2021 及压测实验同尺度，约 2h 量级），**QPS → 资源**；长周期、无 QPS 场景不在本文实现范围。
- **验证**：自建银行系统 + Prometheus 导出，加载 `if_v2021.joblib` 与同一容量公式。
