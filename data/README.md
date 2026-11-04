# 请求集说明

两个请求集都由 `scripts/build_requests.py` 从同一批公开数据生成，**只有每类抽样条数不同**：

| 文件 | 条数 | 每类 | 用于 | 生成命令 |
|---|---|---|---|---|
| `requests.jsonl` | 100 | 25 | 混合流量矩阵（`docs/report-mixed100.md`） | `python scripts/build_requests.py --n 25` |
| `requests_per_dataset.jsonl` | 256 | 64 | 按类别矩阵（`docs/report-per-dataset.md`） | `python scripts/build_requests.py --n 64` |

## 为什么能确认 256 条集就是当时用的那份

抽样流程是「按 prompt 归一化去重 → `random.seed(20261030)` → 洗牌 → 取前 N 条」，
所以 N=64 的结果**必然以 N=25 的结果为前缀**。实测：

```
humaneval 前缀一致: True   gsm8k 前缀一致: True
qasper    前缀一致: True   cmrc2018 前缀一致: True
```

而 `requests.jsonl` 已被验证为**逐字节**可由公开数据重放（见下）。两者前缀一致，
说明源数据的版本与洗牌顺序都没有变化，因此 256 条集同样是可重放的，而不是另一次抽样。

## 数据来源与许可证

| 数据集 | 类别 | 来源 | 许可证 | 原始文件 |
|---|---|---|---|---|
| HumanEval | 代码生成 | [openai/human-eval](https://github.com/openai/human-eval) | MIT | `data/raw/humaneval.jsonl.gz` |
| GSM8K | 数学推理 | [openai/grade-school-math](https://github.com/openai/grade-school-math) | MIT | `data/raw/gsm8k_test.jsonl` |
| QASPER | 长文档问答 | [allenai/qasper](https://huggingface.co/datasets/allenai/qasper)（validation） | CC BY 4.0 | `data/raw/qasper_validation.parquet` |
| CMRC2018 | 中文问答（JDDC 替代） | [hfl/cmrc2018](https://huggingface.co/datasets/hfl/cmrc2018)（validation） | Apache-2.0 | `data/raw/cmrc2018_validation.parquet` |

> **JDDC 2.1 未获取**：需申请授权，当时 HF 上四个镜像均返回 401。按任务书改用可获取的
> CMRC2018，类别标注为「中文问答(客服替代)」，**不声称已获取 JDDC**。

原始数据不随仓库分发，用 `python scripts/fetch_datasets.py` 拉取（约 36 MB）。
仓库只保留**抽样后的请求集**（prompt + 参考答案 + 溯源 id），四类各 25/64 条。

## 可重放性验证

```bash
python scripts/fetch_datasets.py                     # 下载 4 个公开数据集
python scripts/build_requests.py --n 25 --out /tmp/r.jsonl
sha256sum /tmp/r.jsonl data/requests.jsonl           # 应完全一致
```

`requests.jsonl` 的 sha256：`82d242f3dc47b194144254f570c9a6ff9e7cfe3e665e0fd9391d7155812b3b53`

## 字段

```json
{"request_id": "humaneval_000", "dataset": "humaneval", "category": "代码生成",
 "source_id": "<原始数据集里的 id>", "prompt": "<送进模型的输入>", "target": "<参考答案>"}
```

- `prompt` 里**不含**答案；QASPER 用正文段落（最多 40 段、截断到 12k 字符），不是标题拼接。
- `target` 只用于质量评分（`scripts/score2.py`），压测时不会送进服务。
- 中文数据集 `category` 标注了「客服替代」，避免和真正的客服语料混淆。
