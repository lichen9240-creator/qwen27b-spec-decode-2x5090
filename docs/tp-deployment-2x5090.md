# 千问 32B × 双卡 RTX 5090：张量并行部署实录

> **后续说明**：本文是 TP 部署流程的验证记录（用 Qwen3-32B-AWQ 打通链路）。
> 确认 `Qwen/Qwen3.8-27B` 存在后，正式实验已切换到 **Qwen3.8-27B-FP8**，
> 结果见 [`docs/report-mixed100.md`](report-mixed100.md) 与 [`docs/report-per-dataset.md`](report-per-dataset.md)。
> 下文命令与参数对两者通用，只需替换模型路径。

> 环境：AutoDL / Ubuntu 22.04 / 2×RTX 5090 32GB / 驱动 595.71
> 模型：Qwen3-32B-AWQ（官方无 27B 规格，取最接近的 32B，Int4 量化 18GB）
> 版本：vLLM 0.32.0 + torch 2.13.0+cu130 + transformers 5.19.0

---

## 一、部署用到的 Linux 命令

| 命令 | 作用 |
|---|---|
| `nvidia-smi -L`<br>`nvidia-smi --query-gpu=name,memory.total --format=csv` | 确认卡型号、显存、驱动；5090 为 32GB、sm_120 |
| `df -h /root/autodl-tmp` | 看数据盘余量（模型 + 环境要 30GB+） |
| `conda create -p /root/autodl-tmp/env/vllm python=3.12 -y` | 在数据盘建独立环境，不占系统盘 |
| `pip install vllm -i <清华源>` | 装 vLLM；默认轮子已带 CUDA 13，5090 可直接用 |
| `pip install --force-reinstall --no-deps torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0` | 三件套 CUDA 版本串味时，强制对齐到同一版本 |
| `python -c "import torch;print(torch.cuda.get_device_capability(0))"` | 验证卡可用，必须输出 `(12, 0)` |
| `pip install modelscope` | 装国内模型下载器 |
| `export no_proxy=...,modelscope.cn` + `unset http_proxy https_proxy` | 让下载直连，不走代理（3MB/s → 17MB/s） |
| `modelscope download --model Qwen/Qwen3-32B-AWQ --local_dir /root/autodl-tmp/models/...` | 下载权重到数据盘 |
| `nohup <命令> > log 2>&1 &` | 后台启动服务，输出写日志，退出 SSH 不中断 |
| `tail -f log`、`tail -c 500 log` | 看实时/尾部日志；进度条用 `tr '\r' '\n'` 转换才可读 |
| `du -sm 目录` | 核对下载体积是否完整 |
| `ss -ltnp` | 列出监听端口与 PID，配合 `grep :8000` 定位服务 |
| `curl http://127.0.0.1:8000/health` | 健康检查，200 才算起来了 |
| `curl .../v1/models -H "Authorization: Bearer magedu"` | 确认服务名与最大上下文生效 |
| `nvidia-smi --query-compute-apps=pid --format=csv` | 列出占显存的进程，用于清残留 |
| `kill -TERM <pid>` | 优雅停止；不要 `kill -9`（会留 worker 占卡） |

---

## 二、双卡张量并行：初学者怎么上手

### 1. 原理一句话

张量并行（TP）把**每一层的矩阵按注意力头切开**，两张卡各算一半，每层末尾做一次 all-reduce 同步。
所以两个前提：**卡间带宽要高**，且 **TP 数必须整除注意力头数**（32B 是 64 头，2/4/8 都行）。

### 2. 先算账，再启动

| 项 | 计算 | 结果 |
|---|---|---|
| 权重/卡 | 18GB ÷ 2 | ≈9GB |
| KV/卡 | 2×64层×8头×128×2B ×16384 token ÷ 2 | ≈2GB |
| 可用显存 | 32GB × 0.9 | 29GB |

9 + 2 + 约 2GB 激活开销 < 29GB → **放得下，直接 TP=2**，不需要再降精度。

### 3. 四步照抄

**① 环境**（5090 强制要求新 CUDA）

```bash
conda create -p /root/autodl-tmp/env/vllm python=3.12 -y
/root/autodl-tmp/env/vllm/bin/pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple

# 必须验证：输出 (12, 0) 才是能跑 5090 的构建
/root/autodl-tmp/env/vllm/bin/python -c "import torch;print(torch.__version__, torch.cuda.get_device_capability(0))"

# 若报 torch 与 torchaudio 的 CUDA 版本不一致，把三件套对齐（版本号与 vLLM 要求一致）
/root/autodl-tmp/env/vllm/bin/pip install --force-reinstall --no-deps \
  torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**② 模型**（放数据盘）

```bash
modelscope download --model Qwen/Qwen3-32B-AWQ \
  --local_dir /root/autodl-tmp/models/Qwen3-32B-AWQ
du -sm /root/autodl-tmp/models/Qwen3-32B-AWQ   # 应约 18200MB
```

**③ 启动**（和单卡只差一个参数）

```bash
nohup /root/autodl-tmp/env/vllm/bin/vllm serve \
  /root/autodl-tmp/models/Qwen3-32B-AWQ \
  --served-model-name qwen3-32b \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384 \
  --host 127.0.0.1 --port 8000 --api-key magedu \
  > /root/autodl-tmp/logs/vllm_tp2.log 2>&1 &
```

> `--tensor-parallel-size 2` 是唯一的并行开关：写 2 切两张卡，写 1 就是单卡。
> 对比实验时其余参数必须完全一致，否则数字不可比。

**④ 验收三件事**（缺一不可）

- 日志里两个 worker **各加载 9.15 GiB**（不是各自 18GB）→ 切分生效
- `nvidia-smi` 两张卡占用**完全相同**：29762 / 29762 MiB → 负载均衡
- `/health` 返回 200 且 `/v1/models` 里是 `qwen3-32b` → 服务可用

### 4. 实测：TP 到底值不值

输入 512 / 输出 128 token，temperature=0，预热 2 次：

| 配置 | 并发 | TTFT p50/p95 | TPOT | 输出吞吐 |
|---|---|---|---|---|
| TP=1 | 1 | 44.8 / 44.8 ms | 13.2 ms | 74.5 tok/s |
| TP=1 | 8 | 88.1 / 95.2 ms | 13.5 ms | 569.5 tok/s |
| **TP=2** | 1 | **23.4 / 24.9 ms** | **9.4 ms** | **105.0 tok/s** |
| **TP=2** | 8 | **62.9 / 72.5 ms** | 10.2 ms | **751.9 tok/s** |

- 单请求：TTFT 快 **1.91×**，TPOT 快 **1.40×**
- 8 并发吞吐只快 **1.32×**，扩展效率 **0.66**

**结论**：5090 没有 NVLink，all-reduce 全走 PCIe，所以 TP 买的是**延迟**，不是**吞吐**。
要低延迟（对话、Agent）→ 上 TP；要吞吐（批量离线）→ 多副本各占一张卡更划算。

### 5. 五个必踩的坑

1. **旧镜像直接跑**：torch 2.5+cu124 不认 sm_120，报 `no kernel image is available`。必须换 cu128+ 的 torch。
2. **为指定 CUDA 加了 `--extra-index-url`**：torch 可能装成 cu130，torchaudio/torchvision 却是 cu128 → 一 import 就崩。要么别加，要么三件套一起装同一版本。
3. **下载走代理**：只有 3MB/s。给 `no_proxy` 加 `modelscope.cn` 并清空代理变量，速度翻 5 倍。
4. **`pkill -f vllm` 自杀**：自己的 SSH 会话命令行里也包含这串字符，会被一起杀掉。改成按端口反查 PID。
5. **`kill -9` 留残影**：父进程死了 worker 还占着显存。先 `kill -TERM`，再用 `nvidia-smi --query-compute-apps=pid` 清残留。

### 6. 现在怎么用

本地开隧道，然后像调 OpenAI 一样调它：

```bash
# 换成你自己的实例地址与端口（AutoDL 控制台 → 实例 → SSH 登录指令）
ssh -CNg -L 8000:127.0.0.1:8000 root@<你的实例地址> -p <你的端口>

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer magedu" \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

要换模型/换卡，只改三个量：模型路径、`--tensor-parallel-size`、`--max-model-len`（先按 §2 算账）。
