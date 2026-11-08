# 千问 27B × 双 RTX 5090：投机解码实测

在两张消费级 RTX 5090 上（TP=2、无 NVLink），把 **Qwen3.8-27B-FP8** 用 vLLM 0.32.0 起服务，
对 **5 种投机解码路径**做对照压测，回答一个工程问题：

> 这套卡、这个模型、这类流量 —— **投机解码该不该开？开哪个？什么时候开了反而更慢？**

仓库里是**完整可复现的实验**：请求集生成脚本、压测脚本、385 次运行的原始数据、失败现场日志、
两份实验报告，以及一个**不用 GPU 就能跑的一致性校验**（`scripts/reproduce_tables.py`）。

---

## 结论（都有下方数据支撑）

| 问题 | 答案 |
|---|---|
| 低并发谁最快 | **DFlash2 K=6**：C=1 **203.5 tok/s，2.79×** BASE，TPOT 3.99 ms |
| 高并发谁最稳 | **MTP3**：C=8 **666.3 tok/s（1.60×）**，且 KV 容量只掉到 112k |
| 谁是陷阱 | **SUFFIX**：冷启动仅 **1.14×**；"热缓存 2.25×" 是重复请求让后缀树命中了自己上一轮的输出 |
| 代价是什么 | KV 池容量：BASE 207k → MTP3 112k → SUFFIX 102k → DFlash2 58k → DSpark 57k tokens |
| EAGLE-4 | **两条路径都跑不通**，有原文报错、无性能数据（不编造数字） |
| 收益由什么决定 | **任务类型**，不是 K、也不是引擎：接受率 0.381~0.828，跨任务差 **2.17 倍** |

### 混合 100 条真实请求（C=1/4/8，3 次重复中位数）

| 方法 | C=1 | C=4 | C=8 | 加速 1 / 4 / 8 | TPOT C=1 | 接受率 | KV tokens |
|---|---|---|---|---|---|---|---|
| BASE | 72.9 | 237.1 | 417.1 | 1.00 / 1.00 / 1.00 | 13.19 ms | — | 207,209 |
| MTP3 | 156.0 | 437.7 | **666.3** | 2.14 / 1.85 / **1.60** | 5.49 ms | 0.622 | 112,167 |
| **DFLASH2** | **203.5** | **530.3** | 573.3 | **2.79** / **2.24** / 1.37 | **3.99 ms** | 0.568 | 58,391 |
| DSPARK | 170.6 | 385.6 | 530.5 | 2.34 / 1.63 / 1.27 | 4.84 ms | 0.583 | 56,764 |
| SUFFIX（冷启动） | 82.9 | — | — | **1.14** | 11.36 ms | 0.311 | 101,693 |
| SUFFIX（3 次中位数） | 164.0 | 489.0 | 839.7 | 2.25 / 2.06 / 2.01 | 4.80 ms | 0.609 | 101,693 |

![并发对比](figures/fig_real_concurrency.png)

> **SUFFIX 的缓存污染**：同一请求集重复压测时，后缀树缓存了上一轮回复，
> 接受率 R1 0.311 → R2 0.609 → R3 0.719，吞吐 82.9 → 164.0 → 195.3 tok/s。
> **公平值取 R1 冷启动（1.14×）**；中位数只代表"重复流量"这一种特殊场景。
>
> C=4 的冷启动另有 1 次独立补采（新起服务）：**210.4 tok/s、接受率 0.311，约 0.89× BASE —— 比 BASE 还慢**。
> 同一相位里 `_C04_R01_A` 的 376.0 tok/s 不算冷启动：它跑在同一实例 C=1 三轮之后，缓存已被预热
> （接受率已升到 0.659）。冷启动 C=8 没有独立补采，因此留空，不拿预热值充数。

### 按任务类别（4 数据集 × 64 条，C=1/4/8/16/32，3 次重复）

| 类别 | 最优方法 | C=1 加速 | C=8 加速 | C=32 加速 | 接受率 (C=1) |
|---|---|---|---|---|---|
| 代码 HumanEval | **DFlash2 K=6** | **4.17×** | 3.22× | 1.34× | **0.837** |
| 数学 GSM8K | **DFlash2 K=6** | **3.40×** | 2.80× | 1.14× | 0.621 |
| 长文档 QASPER | **SUFFIX** | **1.87×** | 0.73× | 0.93× | 0.526 |
| 中文问答 CMRC2018 | **DFlash2 K=6** | **2.18×** | 1.60× | 1.07× | 0.408 |

![加速比热力图](figures/fig_ds_speedup_heatmap.png)

**同一套 DFlash2，代码任务 4.17×、长文档只有 1.72×** —— 收益差距来自任务本身，不是调参。
长文档是「长输入 + 短输出」，而投机只作用于 decode 段，prefill 一分钱都不省。

### 三条可以直接拿去用的结论

1. **先看任务，再谈投机。** 短输出 / 长输入（RAG、长文档问答）不要期待投机解码：
   QASPER 上 C≥8 起四种方法**全部低于 BASE**（0.68–0.73×）。
2. **高并发会吃掉收益。** 混合流量下 DFlash2 从 C=1 的 2.79× 掉到 C=8 的 1.37×，
   而 MTP3 反而在高并发最稳（1.60×）。**要吞吐选 MTP，要延迟选 DFlash2。**
3. **KV 容量是隐形账单。** DFlash2 的 2.79× 是拿 KV 池从 207k 砍到 58k 换来的
   （约 72%），高并发长上下文时会先撞容量墙。

### 这几条结论**不能**支持的

诚实边界，务必一起读（详见各报告 §局限）：

- **EAGLE-4 没有任何性能数字**。官方路径报 `Model architectures [...] are not supported`，
  harness 路径报 `CUDA error: invalid argument`。仓库保留了完整失败日志，**没有用估计值填充**。
- **C=16/32 方差大**（同一配置两次重复可差 30–100%），高分位应视为区间而非点值。
- **SUFFIX 无法在固定请求集上做真正的重复实验**（缓存效应），只有 1 次冷启动。
- **质量只做了弱验证**：GSM8K 0.80→0.76 是 25 条里 1 条之差，**未做显著性检验**，
  不能据此断言质量退化；HumanEval 只查语法，不是 pass@1。
- **单机、单模型、单次会话**，未做跨机复现。
- **JDDC 2.1 未获取**，用 CMRC2018 替代并标注。

---

## 仓库结构

```
README.md                    本文件：结论 + 边界
REPRODUCE.md                 从零复现手册（环境、模型、取数、压测、汇总、坑）
docs/
  report-mixed100.md         报告 ①：混合 100 条真实请求（5 方法 × C=1/4/8 × 3 重复）
  report-per-dataset.md      报告 ②：按任务类别（4 类 × 64 条 × C=1..32 × 3 重复）
  report-per-dataset.pdf     报告 ② 的 PDF 版
  tp-deployment-2x5090.md    前置：双卡 TP=2 部署实录（先用 Qwen3-32B-AWQ 打通链路）
  environment.txt            实测环境快照（驱动 / torch / vLLM / 拓扑 / 权重体积）
data/
  requests.jsonl             100 条混合请求集（4 类 × 25，去重后固定 seed 抽样）
  requests_per_dataset.jsonl 256 条分类别请求集（4 类 × 64）
  README.md                  来源、许可证、可重放性验证
scripts/
  fetch_datasets.py          下载 4 个公开数据集
  build_requests.py          生成请求集（--n 25 / --n 64）
  run_mixed100.py            压测：每方法起一次服务，逐并发跑 N 次重复
  run_per_dataset.py         压测：按数据集分组 + 瓶颈验证（BNB / fp8 KV / EAGLE）
  make_tables.py             汇总表 + 并发曲线图
  analyze_ds.py              分类别聚合（中位数 / 加速比 / 排名）
  make_figs2.py              硬件与任务类型图
  make_figs_ds.py            分类别三张图
  build_excel.py             生成 runs_all.xlsx（6 个 sheet）
  score2.py                  输出质量评分（字符 F1 / 数值准确率 / 语法）
  reproduce_tables.py        ★ 不需要 GPU：从 CSV 重算全部表并和报告数字对账
results/
  real_runs.csv              385 次运行的 34 字段原始数据（含 phase 列可追溯）
  runs_all.xlsx              汇总工作簿
  per_dataset_summary.md     分类别汇总
  results_summary.md         混合 100 条汇总
  raw_per_request.tgz        24 个逐请求 JSONL（TTFT/ITL/输出长度）
  k-sweep/                   K 值扫描原始记录
  recomputed-tables.md       由 reproduce_tables.py 生成的重算结果
figures/                     6 张图（报告引用）
logs/
  all_logs.tgz               服务日志 + EAGLE-4 失败原文
  logs_v3.tgz                分类别轮的日志（含 kv/bnb 验证）
  eagle_calibration.tgz      EAGLE-4 width-8 校准产物
  k-sweeps/                  K 值扫描的服务日志
```

---

## 快速开始

**3 分钟，不用 GPU**：把报告里的表从原始 CSV 重算一遍并核对。

```bash
git clone https://github.com/lichen9240-creator/qwen27b-spec-decode-2x5090.git
cd qwen27b-spec-decode-2x5090
python scripts/reproduce_tables.py --check
# → 一致性检查：12 项，0 项失败
```

**完整复现**（约 2 张 5090、1 天）：见 **[REPRODUCE.md](REPRODUCE.md)**。
最短路径是四步：

```bash
python scripts/fetch_datasets.py                  # 拉公开数据
python scripts/build_requests.py --n 25           # 生成请求集（与仓库内文件逐字节一致）
python scripts/run_mixed100.py --methods BASE,MTP3,DFLASH2,DSPARK,SUFFIX \
       --concurrency 1,4,8 --reps 3 --phase A     # 起服务压测
python scripts/make_tables.py && python scripts/reproduce_tables.py --check
```

---

## 环境

| 项目 | 值 |
|---|---|
| GPU | 2× RTX 5090 32607 MiB（sm_120），GPU0–GPU1 拓扑 = `SYS`（跨 NUMA，**无 P2P / 无 NVLink**） |
| 驱动 | 580.105.08 |
| 引擎 | vLLM 0.32.0（EAGLE 定制构建 `0.32.0rc1.dev283+eagle.5`） |
| 基础栈 | torch 2.13.0+cu130、transformers 5.19.0、CUDA 13.0 |
| 模型 | Qwen3.8-27B-FP8（29.5 GB，`qwen3_5`，64 层 / 16 层 full attention / 内置 1 层 MTP） |
| 草稿权重 | DFlash2 3.6 GB、DSpark 3.7 GB、EAGLE-4 5.9 GB |
| 服务参数 | `--tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max-model-len 8192` |

完整快照见 [`docs/environment.txt`](docs/environment.txt)。

---

## 数据与许可

请求集来自 HumanEval（MIT）、GSM8K（MIT）、QASPER（CC BY 4.0）、CMRC2018（Apache-2.0），
只保留抽样后的 prompt 与参考答案，来源与许可证见 [`data/README.md`](data/README.md)。
原始数据请用 `scripts/fetch_datasets.py` 自行下载，不随仓库分发。

## 相关

- **[llm-sizing-calculator](https://github.com/lichen9240-creator/llm-sizing-calculator)** ——
  把这批实测点做成了选型计算器的性能回归基线（7 个实测点 MAPE 7.3%）。
- **[eagle3-qwen3-1.7b-repro](https://github.com/lichen9240-creator/eagle3-qwen3-1.7b-repro)** ——
  自训 EAGLE-3 draft 的完整流水线（单卡 4090，2.135×）。

## License

[MIT](LICENSE)。实验数据与报告同样以 MIT 提供；第三方数据集遵循各自许可证。
