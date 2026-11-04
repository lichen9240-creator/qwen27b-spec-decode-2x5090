#!/usr/bin/env python3
"""真实请求集上的投机解码对比：每方法起一次服务，逐并发跑 N 次重复。

记录：逐请求 TTFT/TPOT/ITL/输出长度 + /metrics 接受率 + 双卡硬件指标 + 显存清理快照。
"""
import argparse
import csv
import json
import os
import random
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ---- 路径与密钥：全部可用环境变量覆盖；默认值即本次实验机器上的位置 ----
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("BENCH_ROOT", REPO)
MODEL = os.environ.get("MODEL", "/root/autodl-tmp/models/Qwen3.8-27B-FP8")
VLLM = os.environ.get("VLLM_BIN", "/root/env/vllm/bin/vllm")
PY = os.environ.get("VLLM_PYTHON", "/root/env/vllm/bin/python")
SERVED = os.environ.get("SERVED_NAME", "qwen3.8-27b")
PORT = int(os.environ.get("PORT", "8000"))
KEY = os.environ.get("API_KEY", "magedu")          # 本地临时 key，非敏感凭据
DFLASH = os.environ.get("DRAFT_DFLASH", "/root/autodl-tmp/models/dflash2-27b")
DSPARK = os.environ.get("DRAFT_DSPARK", "/root/autodl-tmp/models/dspark-27b")
EAGLE = os.environ.get("DRAFT_EAGLE4", "/root/autodl-tmp/models/eagle4-27b")
REQUESTS = os.environ.get("REQUESTS_JSONL", os.path.join(REPO, "data", "requests.jsonl"))
E = ROOT                                            # 所有产物写到 BENCH_ROOT 下

METHODS = {
    "BASE": {"spec": None, "extra": []},
    "MTP3": {"spec": {"method": "mtp", "num_speculative_tokens": 3}, "extra": []},
    "SUFFIX": {"spec": {"method": "suffix", "num_speculative_tokens": 4}, "extra": []},
    "DFLASH2": {"spec": {"method": "dflash", "model": DFLASH, "num_speculative_tokens": 6}, "extra": []},
    "DSPARK": {"spec": {"method": "dspark", "model": DSPARK, "num_speculative_tokens": 4},
               "extra": ["--trust-remote-code"]},
    # 需要 EAGLE 定制构建的 vLLM；本仓库实验中该路径失败，见 docs/report-mixed100.md §3
    "EAGLE4H": {"spec": {"method": "eagle_harness", "model": EAGLE, "num_speculative_tokens": 4},
                "extra": [], "vllm": os.environ.get("VLLM_EAGLE_BIN", "/root/env/eagle/bin/vllm"),
                "env": {"VLLM_USE_V2_MODEL_RUNNER": "1"}},
}

FIELDS = ["run_id", "method", "concurrency", "rep", "dataset", "requests", "ok", "failed",
          "output_tok_per_s", "ttft_p50", "ttft_p95", "ttft_p99", "tpot_p50", "tpot_p95",
          "itl_p50", "itl_p95", "itl_p99", "accept_rate", "draft_tokens", "accepted_tokens",
          "avg_in_tokens", "avg_out_tokens", "gpu0_mem_gib", "gpu1_mem_gib", "gpu0_util",
          "gpu1_util", "gpu0_power", "gpu1_power", "tokens_per_watt", "kv_tokens",
          "vram_before_mib", "vram_after_mib", "server_log", "raw_jsonl"]


def http(path, payload=None, timeout=600, stream=False, key=KEY):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path), data=data,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + key})
    return urllib.request.urlopen(request, timeout=timeout)


def gpu_snapshot():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return [int(line.split(",")[1]) for line in out.strip().splitlines() if line]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return []


def _vllm_pids():
    """扫描 /proc，找出所有 vllm 相关进程（不误伤本脚本）。"""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % entry, "rb") as fh:
                cmd = fh.read().decode("utf-8", "replace").replace("\0", " ")
        except (OSError, ValueError):
            continue
        if ("vllm" in cmd or "EngineCore" in cmd) and "run_real.py" not in cmd:
            pids.append(int(entry))
    return pids


def clean_gpus(wait=240):
    """彻底清理：杀 vllm 系进程 → 等到连续两次 <500MiB 才算干净。"""
    before = gpu_snapshot()
    stable = 0
    deadline = time.time() + wait
    while time.time() < deadline:
        used = max(gpu_snapshot() or [0])
        if used < 500:
            stable += 1
            if stable >= 2:
                return before
        else:
            stable = 0
            for pid in _vllm_pids():
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        time.sleep(5)
    return before


class Sampler(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.stop_flag, self.proc = path, threading.Event(), None

    def run(self):
        cmd = ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,"
               "memory.used,power.draw,temperature.gpu,clocks.sm",
               "--format=csv,noheader,nounits", "-l", "1"]
        with open(self.path, "w") as fh:
            fh.write("timestamp,index,sm_util,mem_util,mem_used,power,temp,clock\n")
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
            for line in self.proc.stdout:
                if self.stop_flag.is_set():
                    break
                fh.write(",".join(x.strip() for x in line.split(",")) + "\n")

    def stop(self):
        self.stop_flag.set()
        if self.proc:
            try:
                self.proc.terminate()
            except OSError:
                pass


def counters():
    try:
        with http("/metrics", timeout=5) as resp:
            text = resp.read().decode()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if "spec_decode" in line and not line.startswith("#"):
            name = line.split("{")[0].split(" ")[0]
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                pass
    return out


def one_request(item, max_tokens):
    payload = {"model": SERVED, "messages": [{"role": "user", "content": item["prompt"]}],
               "max_tokens": max_tokens, "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    started = time.perf_counter()
    ttft, itls, last, deltas, usage = None, [], None, 0, {}
    try:
        with http("/v1/chat/completions", payload, stream=True) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except ValueError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                content = (choices[0].get("delta") or {}).get("content")
                if not content:
                    continue
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - started
                elif last is not None:
                    itls.append(now - last)
                last = now
                deltas += 1
        elapsed = time.perf_counter() - started
    except Exception as exc:
        return {"request_id": item["request_id"], "dataset": item["dataset"], "ok": False,
                "error": str(exc)[:200]}
    return {"request_id": item["request_id"], "dataset": item["dataset"], "ok": True,
            "ttft": ttft, "e2e": elapsed,
            "out_tokens": usage.get("completion_tokens") or deltas,
            "in_tokens": usage.get("prompt_tokens") or 0,
            "itl": itls}


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def start_server(method, log_path):
    spec = METHODS[method]
    binary = spec.get("vllm", VLLM)
    cmd = [binary, "serve", MODEL, "--served-model-name", SERVED,
           "--tensor-parallel-size", "2", "--gpu-memory-utilization", "0.9",
           "--max-model-len", "8192", "--host", "127.0.0.1", "--port", str(PORT),
           "--api-key", KEY] + spec["extra"]
    if spec["spec"]:
        cmd += ["--speculative-config", json.dumps(spec["spec"])]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env.update(spec.get("env", {}))
    log = open(log_path, "w")
    log.write("$ " + " ".join(cmd) + "\n\n")
    log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    return proc, log, cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="BASE")
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--phase", default="A")
    args = parser.parse_args()

    for sub in ("logs", "metrics", "raw_data", "results", "configs"):
        os.makedirs(os.path.join(E, sub), exist_ok=True)
    requests_list = [json.loads(l) for l in open(os.path.join(E, "requests.jsonl"))][:args.limit]
    csv_path = os.path.join(E, "results", "real_runs.csv")
    new_file = not os.path.isfile(csv_path)
    date = time.strftime("%Y%m%d")

    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        if method not in METHODS:
            print("未知方法", method)
            continue
        before = clean_gpus()
        log_path = os.path.join(E, "logs", "%s_%s_server.log" % (date, method))
        proc, log, cmd = start_server(method, log_path)
        with open(os.path.join(E, "configs", "%s_%s.json" % (date, method)), "w") as fh:
            json.dump({"method": method, "cmd": cmd, "spec": METHODS[method]["spec"]}, fh, indent=2)

        started = time.time()
        ok = False
        while time.time() - started < 900:
            if proc.poll() is not None:
                break
            try:
                with http("/health", timeout=3) as resp:
                    if resp.status == 200:
                        ok = True
                        break
            except Exception:
                pass
            time.sleep(5)
        if not ok:
            print("!! %s 启动失败 %.0fs" % (method, time.time() - started))
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            log.close()
            before = clean_gpus()
            continue

        text = open(log_path, errors="replace").read()
        kv = re.search(r"GPU KV cache size:\s*([\d,]+)", text)
        kv_tokens = int(kv.group(1).replace(",", "")) if kv else 0
        print("== %s 就绪 %.0fs KV=%s" % (method, time.time() - started, kv_tokens), flush=True)

        for conc in [int(c) for c in args.concurrency.split(",")]:
            for rep in range(1, args.reps + 1):
                run_id = "%s_%s_C%02d_R%02d_%s" % (date, method, conc, rep, args.phase)
                sampler = Sampler(os.path.join(E, "metrics", run_id + "_gpu.csv"))
                sampler.start()
                time.sleep(2)
                c0 = counters()
                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=conc) as pool:
                    results = list(pool.map(lambda it: one_request(it, args.max_tokens), requests_list))
                wall = time.perf_counter() - t0
                sampler.stop()
                c1 = counters()

                draft = sum(v for k, v in c1.items() if "draft" in k) - sum(v for k, v in c0.items() if "draft" in k)
                accept = (sum(v for k, v in c1.items() if "accepted" in k and "per_pos" not in k)
                          - sum(v for k, v in c0.items() if "accepted" in k and "per_pos" not in k))
                good = [r for r in results if r.get("ok")]
                out_tokens = sum(r["out_tokens"] for r in good)
                ttfts = [r["ttft"] for r in good if r.get("ttft")]
                tpots = [(r["e2e"] - r["ttft"]) / (r["out_tokens"] - 1) * 1000
                         for r in good if r.get("ttft") and r["out_tokens"] > 1]
                itls = [v * 1000 for r in good for v in r["itl"]]

                sample = os.path.join(E, "metrics", run_id + "_gpu.csv")
                mems, utils, powers = [], [], []
                with open(sample) as fh:
                    for row in csv.DictReader(fh):
                        try:
                            mems.append((int(row["index"]), float(row["mem_used"]) / 1024))
                            utils.append((int(row["index"]), float(row["sm_util"])))
                            powers.append((int(row["index"]), float(row["power"])))
                        except (KeyError, ValueError):
                            continue
                g0m = max([v for i, v in mems if i == 0], default=0)
                g1m = max([v for i, v in mems if i == 1], default=0)
                g0u = sum(v for i, v in utils if i == 0) / max(1, len([1 for i, _ in utils if i == 0]))
                g1u = sum(v for i, v in utils if i == 1) / max(1, len([1 for i, _ in utils if i == 1]))
                g0p = sum(v for i, v in powers if i == 0) / max(1, len([1 for i, _ in powers if i == 0]))
                g1p = sum(v for i, v in powers if i == 1) / max(1, len([1 for i, _ in powers if i == 1]))
                thr = out_tokens / wall if wall else 0
                after = gpu_snapshot()

                row = {
                    "run_id": run_id, "method": method, "concurrency": conc, "rep": rep,
                    "dataset": "mixed100", "requests": len(results), "ok": len(good),
                    "failed": len(results) - len(good),
                    "output_tok_per_s": round(thr, 1),
                    "ttft_p50": round(pct(ttfts, .5) * 1000, 1) if ttfts else None,
                    "ttft_p95": round(pct(ttfts, .95) * 1000, 1) if ttfts else None,
                    "ttft_p99": round(pct(ttfts, .99) * 1000, 1) if ttfts else None,
                    "tpot_p50": round(pct(tpots, .5), 2) if tpots else None,
                    "tpot_p95": round(pct(tpots, .95), 2) if tpots else None,
                    "itl_p50": round(pct(itls, .5), 2) if itls else None,
                    "itl_p95": round(pct(itls, .95), 2) if itls else None,
                    "itl_p99": round(pct(itls, .99), 2) if itls else None,
                    "accept_rate": round(accept / draft, 4) if draft else None,
                    "draft_tokens": int(draft), "accepted_tokens": int(accept),
                    "avg_in_tokens": round(sum(r["in_tokens"] for r in good) / max(1, len(good)), 1),
                    "avg_out_tokens": round(out_tokens / max(1, len(good)), 1),
                    "gpu0_mem_gib": round(g0m, 2), "gpu1_mem_gib": round(g1m, 2),
                    "gpu0_util": round(g0u, 1), "gpu1_util": round(g1u, 1),
                    "gpu0_power": round(g0p, 1), "gpu1_power": round(g1p, 1),
                    "tokens_per_watt": round(thr / (g0p + g1p), 3) if (g0p + g1p) else None,
                    "kv_tokens": kv_tokens,
                    "vram_before_mib": "/".join(str(x) for x in before),
                    "vram_after_mib": "/".join(str(x) for x in after),
                    "server_log": log_path,
                    "raw_jsonl": os.path.join(E, "raw_data", run_id + ".jsonl"),
                }
                with open(row["raw_jsonl"], "w") as fh:
                    for item in results:
                        fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                with open(csv_path, "a", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=FIELDS)
                    if new_file:
                        writer.writeheader()
                        new_file = False
                    writer.writerow(row)
                print("  C=%-2d R%d ok=%d thr=%.1f ttft50=%s tpot50=%s acc=%s"
                      % (conc, rep, len(good), thr, row["ttft_p50"], row["tpot_p50"],
                         row["accept_rate"]), flush=True)

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        log.close()
        time.sleep(5)
        after = clean_gpus()
        print("  %s 结束，清理后显存 %s MiB" % (method, after), flush=True)


if __name__ == "__main__":
    random.seed(20261030)
    main()
