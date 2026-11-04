# 请求集清单

| 数据集 | 类别 | 来源 | 许可证 | 条数 |
|---|---|---|---|---|
| HumanEval | 代码生成 | openai/human-eval (GitHub raw) | MIT | 64 |
| GSM8K | 数学推理 | openai/grade-school-math test.jsonl | MIT | 64 |
| QASPER | 长文档问答 | allenai/qasper validation parquet | CC BY 4.0 | 64 |
| CMRC2018 | 中文问答（JDDC 替代） | hfl/cmrc2018 validation parquet | Apache-2.0 | 64 |

> JDDC 2.1 需申请，且 HF 上四个镜像均返回 401；按任务书改用可获取的中文问答集 CMRC2018，类别标注为「中文问答(客服替代)」。

抽样规则：先按 prompt 归一化去重，再随机抽样（seed=20261030），每类 64 条。

> 本文件由 `python scripts/build_requests.py --n 64` 生成，对应 `requests_per_dataset.jsonl`。
> 混合 100 条集用 `--n 25`（对应 `requests.jsonl`），来源与许可证完全相同，只有每类条数不同；
> 两个集合的关系见 [README.md](README.md)。
