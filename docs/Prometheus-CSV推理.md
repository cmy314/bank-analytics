# Prometheus 导出 CSV → IF 离线推理

压测结束后，用 **一条命令** 判断孤立森林是否识别异常。

## 1. 前置条件

- 已用 **四维**（QPS / RT / CPU / Memory）重新训练 `if_v2021.joblib`（旧五维模型需重跑 `v2021-merge-train`）。
- Grafana 或 Prometheus 已采集 `transaction-service`（或目标服务）指标。

## 2. Grafana 导出宽表

在 Grafana 新建 Table 面板，添加 4 条查询（示例）：

| 列名（导出后重命名） | PromQL 示例 |
|---------------------|-------------|
| `timestamp` / `Time` | 使用 Grafana 时间列 |
| `throughput_total` | `bank:http_qps:rate1m{application="transaction-service"}` |
| `rt` | `bank:http_latency_p95:5m{application="transaction-service"}` |
| `instance_cpu_usage` | `avg(process_cpu_usage{application="transaction-service"}) * 100` |
| `instance_memory_usage` | `sum(jvm_memory_used_bytes{application="transaction-service",area="heap"}) / sum(jvm_memory_max_bytes{application="transaction-service",area="heap"}) * 100` |

导出为 CSV，保证时间范围覆盖压测窗口（建议 1min 步长）。

脚本会自动：

- 秒 → 毫秒（RT）
- 0–1 比例 → 百分比（CPU/内存）

## 3. 运行推理

```bash
cd analytics
python infer_from_joblib.py \
  --csv output/prometheus_txn.csv \
  --model output/v2021/if_v2021.joblib \
  --out output/infer_alerts.csv
```

或：

```bash
python -m bank_analytics v2021-infer --csv ... --model ... --out ...
```

## 4. 输出列

| 列 | 含义 |
|----|------|
| `if_pred` | sklearn IF：-1 异常 / 1 正常 |
| `if_score` | 决策函数，越低越异常 |
| `if_alert` | `if_pred == -1` |
| `rule_alert` | CPU 或 RT 超过训练集 P95 阈值 |
| `combined_alert` | IF 或 P95 任一为 1 |
| `p95_cpu_thr` / `p95_rt_thr` | 训练阶段写入 joblib 的阈值 |

终端会打印告警行数；`combined_alert=1` 的行即论文截图用异常时段。

## 5. 与 Prometheus 实时规则的关系

- **在线**：`monitoring/prometheus/rules.yml`（P95 延迟、CPU、QPS 突增等）。
- **离线**：本脚本用同一四维特征 + 训练期 IF/P95，适合压测后批量对照。

两者互补：答辩可先展示 Grafana 告警，再跑本脚本给出 IF 得分表。
