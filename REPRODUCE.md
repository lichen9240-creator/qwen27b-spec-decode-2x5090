# 复现手册

> 目标机器：单机 **2× RTX 5090（32 GB）**，Linux；本文按实际跑通的顺序写，
> 包含所有踩过的坑及其修复。全程约 **1 天**（环境 2 h + 权重 1.5 h + 压测 5 h + 汇总 0.5 h）。
>
> 只想校验数据而不想开卡：直接跳到 [§6](#6-不开卡也能做的校验)，不需要 GPU。

---

## 0. 前置检查

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# 期望：2× NVIDIA GeForce RTX 5090, 32607 MiB, 580.105.08
python -c "import torch;print(torch.cuda.get_device_capability(0))"   # 必须是 (12, 0)

nvidia-smi topo -m        # 观察 GPU0-GPU1：本机是 SYS（跨 NUMA，无 NVLink / 无 P2P）
df -h /root               # 权重 + 草稿 + 环境约需 60 GB
nproc                     # 本机 208 逻辑核；压测端不会成为瓶颈
```

> RTX 5090 是 sm_120，**必须** cu128 及以上的 torch/vLLM 构建，否则报
> `no kernel image is available for execution on the device`。

---

## 1. 环境

### 1.1 主环境（BASE / MTP / DFlash2 / DSpark / SUFFIX）

```bash
conda create -y -p /root/env/vllm python=3.12
/root/env/vllm/bin/pip install -U pip
/root/env/vllm/bin/pip install vllm==0.32.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 三件套必须同一 CUDA 版本；不一致就整组重装
/root/env/vllm/bin/pip install --force-reinstall --no-deps \
  torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0

/root/env/vllm/bin/python -c \
  "import torch,vllm;print(torch.__version__, torch.cuda.get_device_capability(0), vllm.__version__)"
# 期望：2.13.0+cu130 (12, 0) 0.32.0

/root/env/vllm/bin/vllm --version
```

**SUFFIX 的额外依赖**（不加会在起服务时报 `Arctic Inference is required`）：

```bash
/root/env/vllm/bin/pip install arctic-inference==0.1.1
```

### 1.2 EAGLE 环境（可选，本实验两条路径均失败）

EAGLE-4 需要官方定制构建，装在独立 venv 里，避免污染主环境：

```bash
python -m venv /root/env/eagle
/root/env/eagle/bin/pip install -U pip
/root/env/eagle/bin/pip install --index-url <官方 wheel 源> vllm==0.32.0rc1.dev283+eagle.5
/root/env/eagle/bin/python -c "from vllm.spec_decode import eagle_harness; print(eagle_harness.missing_hooks())"
# 期望输出 NONE，说明引擎侧 hook 齐了（本实验仍卡在 GPU 内核执行，见报告 §3）
```

> 在原生 vLLM 0.32.0 上，`eagle-harness 0.1.4` 的自检会直接禁用自己
> （`lacks 8 of the hooks it needs`）。装官方定制构建后 hook 齐了，但起服务仍报
> `CUDA error: invalid argument`——**这就是本实验没有 EAGLE 数字的原因**。

---

## 2. 模型与草稿权重

四个权重都要，放在数据盘（不要放系统盘，会满）：

```bash
export HF_ENDPOINT=https://hf-mirror.com     # 国内直连不通时
mkdir -p /root/models && cd /root/models

# target：约 29.5 GB
huggingface-cli download Qwen/Qwen3.8-27B-FP8 --local-dir Qwen3.8-27B-FP8

# draft 权重（按你拿到的发布地址替换；本实验用的是三个 27B 级草稿）
#   DFlash2 3.6 GB、DSpark 3.7 GB、EAGLE-4 5.9 GB
```

校验体积（`docs/environment.txt` 里的实测值）：

| 目录 | 体积 |
|---|---|
| `Qwen3.8-27B-FP8` | 29,460 MB |
| `dflash2-27b` | 3,671 MB |
| `dspark-27b` | 3,793 MB |
| `eagle4-27b` | 5,892 MB |

需要的话用环境变量告诉脚本权重在哪（不设就用下面的默认值）：

```bash
export MODEL=/root/models/Qwen3.8-27B-FP8
export DRAFT_DFLASH=/root/models/dflash2-27b
export DRAFT_DSPARK=/root/models/dspark-27b
export DRAFT_EAGLE4=/root/models/eagle4-27b
export VLLM_BIN=/root/env/vllm/bin/vllm
```

---

## 3. 请求集

```bash
cd <repo>
pip install pandas pyarrow            # build_requests.py 读 parquet 需要
pip install matplotlib openpyxl       # 出图与生成 xlsx 需要（不做图可跳过）
python scripts/fetch_datasets.py      # HumanEval / GSM8K / QASPER / CMRC2018，约 36 MB

python scripts/build_requests.py --n 25    # → data/requests.jsonl（100 条）
python scripts/build_requests.py --n 64    # → data/requests_per_dataset.jsonl（256 条）
```

`--n 25` 的输出与仓库内 `data/requests.jsonl` **逐字节一致**（sha256 `82d242f3…`），
可作为数据管线的自检。两个集合的关系与来源见 [`data/README.md`](data/README.md)。

---

## 4. 压测

脚本会自动：起服务 → 等 `/health` 就绪（最多 900 s）→ 每格跑 N 次重复 → 采集
`nvidia-smi` 逐秒指标 → 从 `/metrics` 差分算接受率 → 杀进程并确认显存归零 → 写 CSV。

```bash
cd <repo>
export BENCH_ROOT=/root/bench        # 产物目录：logs/ metrics/ results/ configs/ raw_data/
mkdir -p $BENCH_ROOT
```

### 4.1 混合 100 条矩阵（报告 ①，约 1.5 h）

```bash
python scripts/run_mixed100.py \
  --methods BASE,MTP3,DFLASH2,DSPARK,SUFFIX \
  --concurrency 1,4,8 --reps 3 --phase A
```

### 4.2 按类别矩阵（报告 ②，约 5 h）

```bash
export REQUESTS_JSONL=$PWD/data/requests_per_dataset.jsonl
python scripts/run_per_dataset.py \
  --methods BASE,MTP3,DFLASH2,DSPARK,SUFFIX \
  --datasets humaneval,gsm8k,qasper,cmrc2018 \
  --concurrency 1,4,8,16,32 --reps 3 --phase DS --limit 300
```

### 4.3 瓶颈验证（可选，约 1 h）

```bash
# prefill 分块假设：调 max_num_batched_tokens
python scripts/run_per_dataset.py --methods BASE,BASE_BNB4K,BASE_BNB8K,DFLASH2_BNB8K \
  --datasets qasper --concurrency 32 --reps 2 --phase BNB --limit 300

# KV 容量假设：fp8 KV cache（本环境启动失败，属预期结果）
python scripts/run_per_dataset.py --methods BASE_FP8KV,DFLASH2_FP8KV \
  --datasets qasper --concurrency 32 --reps 2 --phase KV --limit 300
```

### 4.4 硬件与能效（报告 ① §6）

```bash
python scripts/run_mixed100.py --methods BASE,MTP3,DFLASH2,DSPARK,SUFFIX \
  --concurrency 4 --reps 1 --phase HW
```

### 4.5 输出质量（可选，需 GPU，约 40 min）

`max_tokens=512` 跑一遍，再用 `score2.py` 评分：

```bash
python scripts/run_mixed100.py --methods BASE,MTP3,DFLASH2 \
  --concurrency 1 --reps 1 --phase Q --max-tokens 512     # 同时生成 raw_data/*.jsonl
python scripts/score2.py                                  # → 质量分数
```

### 4.6 可用参数

| 变量 | 默认 | 说明 |
|---|---|---|
| `BENCH_ROOT` | 仓库根目录 | 所有产物的落盘位置 |
| `REQUESTS_JSONL` | `data/requests.jsonl` | 用哪个请求集 |
| `MODEL` / `VLLM_BIN` | 见 §2 | target 模型与 vLLM 可执行文件 |
| `DRAFT_DFLASH` / `DRAFT_DSPARK` / `DRAFT_EAGLE4` | 见 §2 | 三个草稿权重 |
| `VLLM_EAGLE_BIN` | `/root/env/eagle/bin/vllm` | EAGLE 定制构建 |
| `SERVED_NAME` / `PORT` / `API_KEY` | `qwen3.8-27b` / `8000` / `magedu` | 服务名、端口、本地 key |

服务端固定参数（写死在 `start_server` 里，改对比口径时**必须一起改**）：
`--tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max-model-len 8192`；
请求口径：`temperature=0, max_tokens=128, enable_thinking=False`。

---

## 5. 汇总

```bash
python scripts/make_tables.py      # → results/summary_table.md + figures/fig_real_concurrency.png
python scripts/analyze_ds.py       # → results/per_dataset_summary.md（分类别中位数/加速比/排名）
python scripts/make_figs2.py       # → figures/fig_hw_efficiency.png、fig_accept_by_task.png
python scripts/make_figs_ds.py     # → 分类别三张图
python scripts/build_excel.py      # → results/runs_all.xlsx（6 个 sheet）
python scripts/reproduce_tables.py # → results/recomputed-tables.md + 一致性检查
```

> 这些脚本直接读 `results/real_runs.csv`，不依赖 GPU。若用 `BENCH_ROOT` 跑到了别处，
> 先把 CSV 拷回 `results/real_runs.csv`（或设 `RUNS_CSV` 指向它）。

---

## 6. 不开卡也能做的校验

```bash
python scripts/reproduce_tables.py --check
```

它做两件事：从 `results/real_runs.csv` 重算混合 100 条与四个类别的全部中位数表；
把 12 个头条数字与 README/报告里引用的值对账（1% 容差）。期望输出：

```
A    行数 45（期望 45）OK
DS   行数 300（期望 300）OK
mixed100  DFLASH2 C=1  重算   203.5 / 报告   203.5  偏差 0.00% OK
...
一致性检查：12 项，0 项失败
```

`run_id` 里带 `phase` 字段，可以逐行回溯到具体日志与配置：

| phase | 含义 |
|---|---|
| `A` | 混合 100 条，C=1/4/8，3 重复 |
| `DS` | 4 数据集 × 64 条，C=1/4/8/16/32，3 重复 |
| `HW` | C=4 硬件指标补采 |
| `Q` | 质量评分（max_tokens=512） |
| `BOT` | 按任务类型探测（DFLASH2 / C=1 / 25 条/类） |
| `KV` / `BNB` | 瓶颈验证（fp8 KV / max_num_batched_tokens） |

---

## 7. 预期结果（自查用）

混合 100 条（`phase A`，中位数）：

| 方法 | C=1 | C=4 | C=8 |
|---|---|---|---|
| BASE | 72.9 | 237.1 | 417.1 |
| MTP3 | 156.0 | 437.7 | 666.3 |
| DFLASH2 | 203.5 | 530.3 | 573.3 |
| DSPARK | 170.6 | 385.6 | 530.5 |
| SUFFIX（R1 冷启动） | 82.9 | 210.4 | — |

**容差建议**：绝对值允许 ±10%（消费者卡批次、驱动、后台进程都会影响）；
**加速比**应落在 ±8% 内。差异超出时优先怀疑：GPU 拓扑不是 `SYS`、`--max-model-len` 不一致、
SUFFIX 没冷启动、或者服务没干净重启（显存残留）。

---

## 8. 踩过的坑

**环境**
1. **torch 与 torchaudio / torchvision 的 CUDA 版本串味** → 一 import 就崩。三件套必须整组装同一版本。
2. **旧镜像的 torch 2.5+cu124 不认 sm_120** → `no kernel image is available`，必须 cu128+。
3. **`pkill -f vllm` 会杀掉自己**（SSH 命令行里也含这串字符）。脚本改成扫 `/proc` 按 cmdline 匹配
   并排除自身（`_vllm_pids`），再按显存归零双确认（`clean_gpus`）。
4. **`kill -9` 留残影**：父进程死了 worker 还占显存。用 `kill -TERM` + 轮询显存。

**服务**
5. **SUFFIX 报 `Arctic Inference is required`** → 装 `arctic-inference==0.1.1`。
6. **DSpark 要 `--trust-remote-code`**，否则权重加载失败。
7. **fp8 KV cache 起不来**：FlashInfer 报 `requires GPUs with sm75 or higher`（本环境的反向误判），
   所以 KV 假设只能靠 BASE 的反例间接否掉。
8. **EAGLE-4 两条路径都失败**：官方路径架构不被接受；harness 路径校准 width-8 成功、
   width-16 失败，起服务报 `CUDA error: invalid argument`（换 attention backend 相同）。

**测量**
9. **`nvidia-smi -l 1` 走管道会被缓冲**，导致主矩阵 22/27 行显存字段缺失。
   改成每秒独立采样写 CSV，并在 C=4 单独补采（`phase HW`）。
10. **`utilization.memory` 不是 HBM 带宽**，是"显存忙的时间占比"。报告里没有声称带宽利用率。
11. **SUFFIX 在固定请求集上会被自己的缓存"喂分"**：R1 冷启动才是公平值，务必把 R1 单独报出来。

**压测端**
12. **`--limit` 必须显式给**：默认 100 会截断 256 条的请求集，导致每类只剩 25 条。
13. **Python 线程池不是瓶颈**：208 核机器上 C=32 的客户端开销可忽略；换小机器要先确认。

---

## 9. 时间与成本（AutoDL 2×5090）

| 阶段 | 时长 |
|---|---|
| 环境 + 依赖 | 2 h |
| 下载 4 个权重（约 42 GB） | 1.5 h |
| 混合 100 条矩阵 | 1.5 h |
| 按类别矩阵（含 5 次重启服务） | 5 h |
| 瓶颈验证 + 质量评分 | 1.5 h |
| 汇总与图 | 0.5 h |
| **合计** | **约 12 h**（加上调试约 1 天） |

按当时 2×5090 约 ¥6/h 计，单轮完整复现约 **¥70–80**。
