#!/usr/bin/env python3
"""把所有实验（mixed + per-dataset + HW/LC/Q）合并成一个 Excel 工作簿。"""
import csv
import os
import statistics
from collections import defaultdict

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = os.environ.get("BENCH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.environ.get("RUNS_CSV", os.path.join(ROOT, "results", "real_runs.csv"))
OUT = os.environ.get("XLSX_OUT", os.path.join(ROOT, "results", "runs_all.xlsx"))
DATASETS = ["humaneval", "gsm8k", "qasper", "cmrc2018"]
LABEL = {"humaneval": "代码", "gsm8k": "数学", "qasper": "长文档", "cmrc2018": "中文问答",
         "all": "混合", "mixed100": "混合100"}
HEADER_FILL = PatternFill("solid", fgColor="DDEBF7")
METHODS = ["BASE", "MTP3", "DFLASH2", "DSPARK", "SUFFIX"]


def phase(row):
    parts = row["run_id"].split("_")
    for token in (parts[-1], parts[-2] if len(parts) >= 2 else ""):
        if token in ("A", "HW", "LC", "Q", "BOT", "DS", "PROBE"):
            return token
    return "?"


rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
for row in rows:
    row["phase"] = phase(row)
print("总行数", len(rows))

wb = Workbook()

# ---------- README ----------
ws = wb.active
ws.title = "README"
notes = [
    ["实验汇总（Qwen3.8-27B-FP8 / 2×RTX 5090 / TP=2 / vLLM 0.32.0）"],
    [""],
    ["sheet", "含义"],
    ["runs_all", "每次运行一行，34 列原始指标（含 phase 列）"],
    ["per_dataset", "按 类别 × 方法 × 并发 聚合（新数据，64 条/类，3 次重复，中位数）"],
    ["mixed", "旧的混合 100 条矩阵（5 方法 × C=1/4/8 × 3 重复）"],
    ["hardware", "C=4 硬件指标复采（SM 利用率/功耗/显存/每瓦 token）"],
    ["comparison", "新旧两套口径的加速比对照"],
    [""],
    ["口径说明"],
    ["phase=A", "混合 100 条（旧）"],
    ["phase=DS", "按类别 64 条（新，本次主实验）"],
    ["phase=BOT", "早期单方法探测（DFLASH2, C=1, 25 条/类）"],
    ["phase=HW", "硬件指标复采（C=4，100 条混合）"],
    ["phase=LC", "长上下文（QASPER，C=1/4，2 次重复）"],
    ["phase=Q", "质量评估（max_tokens=512，100 条混合）"],
    [""],
    ["注意"],
    ["SUFFIX 有缓存效应：同一请求集重复压测时接受率会升高，冷启动值取 rep=1"],
    ["QASPER 输入约 2900 token，C=16/32 时 KV 可能成为瓶颈（见报告）"],
]
for line in notes:
    ws.append(line)
ws["A1"].font = Font(bold=True, size=13)
ws.column_dimensions["A"].width = 18
ws.column_dimensions["B"].width = 90

# ---------- runs_all ----------
ws = wb.create_sheet("runs_all")
fields = list(rows[0].keys())
ws.append(fields)
for row in rows:
    ws.append([row.get(f) for f in fields])
for cell in ws[1]:
    cell.font = Font(bold=True)
    cell.fill = HEADER_FILL
ws.freeze_panes = "A2"
ws.auto_filter.ref = ws.dimensions
for idx in range(1, len(fields) + 1):
    ws.column_dimensions[get_column_letter(idx)].width = 14

# ---------- per_dataset ----------
ws = wb.create_sheet("per_dataset")
ws.append(["类别", "方法", "并发", "n", "吞吐 tok/s", "吞吐标准差", "TTFT p50 ms",
           "TPOT p50 ms", "接受率", "KV tokens", "相对 BASE 加速"])
buckets = defaultdict(list)
for row in rows:
    if row["phase"] != "DS":
        continue
    try:
        buckets[(row["dataset"], row["method"], int(row["concurrency"]))].append(row)
    except (TypeError, ValueError):
        continue


def med(items, key):
    vals = []
    for item in items:
        try:
            if item.get(key):
                vals.append(float(item[key]))
        except ValueError:
            pass
    return statistics.median(vals) if vals else None


for dataset in DATASETS:
    for method in METHODS:
        for conc in (1, 4, 8, 16, 32):
            items = buckets.get((dataset, method, conc))
            if not items:
                continue
            thr = med(items, "output_tok_per_s")
            vals = [float(i["output_tok_per_s"]) for i in items if i.get("output_tok_per_s")]
            base_thr = med(buckets.get((dataset, "BASE", conc), []), "output_tok_per_s")
            ws.append([LABEL[dataset], method, conc, len(items),
                       round(thr, 1) if thr else None,
                       round(statistics.pstdev(vals), 2) if len(vals) > 1 else 0,
                       round(med(items, "ttft_p50") or 0, 1),
                       round(med(items, "tpot_p50") or 0, 2),
                       round(med(items, "accept_rate") or 0, 4),
                       int(med(items, "kv_tokens") or 0),
                       round(thr / base_thr, 2) if thr and base_thr else None])
for cell in ws[1]:
    cell.font = Font(bold=True)
    cell.fill = HEADER_FILL
ws.freeze_panes = "C2"
for idx in range(1, 12):
    ws.column_dimensions[get_column_letter(idx)].width = 13

# ---------- mixed ----------
ws = wb.create_sheet("mixed")
ws.append(["方法", "并发", "n", "吞吐 tok/s", "TTFT p50", "TPOT p50", "接受率", "加速比"])
mb = defaultdict(list)
for row in rows:
    if row["phase"] != "A":
        continue
    try:
        mb[(row["method"], int(row["concurrency"]))].append(row)
    except (TypeError, ValueError):
        continue
for conc in (1, 4, 8):
    base_thr = med(mb.get(("BASE", conc), []), "output_tok_per_s")
    for method in METHODS:
        items = mb.get((method, conc))
        if not items:
            continue
        thr = med(items, "output_tok_per_s")
        ws.append([method, conc, len(items), round(thr, 1) if thr else None,
                   round(med(items, "ttft_p50") or 0, 1),
                   round(med(items, "tpot_p50") or 0, 2),
                   round(med(items, "accept_rate") or 0, 4) if med(items, "accept_rate") else None,
                   round(thr / base_thr, 2) if thr and base_thr else None])
for cell in ws[1]:
    cell.font = Font(bold=True)
    cell.fill = HEADER_FILL
for idx in range(1, 9):
    ws.column_dimensions[get_column_letter(idx)].width = 13

# ---------- hardware ----------
ws = wb.create_sheet("hardware")
ws.append(["方法", "吞吐 tok/s", "SM 卡0 %", "SM 卡1 %", "功耗 卡0 W", "功耗 卡1 W",
           "显存/卡 GiB", "每瓦 token"])
seen = set()
for row in rows:
    if row["phase"] != "HW" or row["method"] in seen:
        continue
    seen.add(row["method"])
    ws.append([row["method"], row["output_tok_per_s"], row["gpu0_util"], row["gpu1_util"],
               row["gpu0_power"], row["gpu1_power"], row["gpu0_mem_gib"],
               row["tokens_per_watt"]])
for cell in ws[1]:
    cell.font = Font(bold=True)
    cell.fill = HEADER_FILL
for idx in range(1, 9):
    ws.column_dimensions[get_column_letter(idx)].width = 14

# ---------- comparison ----------
ws = wb.create_sheet("comparison")
ws.append(["方法", "混合100 C=1", "按类别均值 C=1", "混合100 C=8", "按类别均值 C=8"])
for method in METHODS:
    def speed(phase_name, conc):
        if phase_name == "A":
            base_thr = med(mb.get(("BASE", conc), []), "output_tok_per_s")
            thr = med(mb.get((method, conc), []), "output_tok_per_s")
        else:
            sp = []
            for dataset in DATASETS:
                base_thr = med(buckets.get((dataset, "BASE", conc), []), "output_tok_per_s")
                thr = med(buckets.get((dataset, method, conc), []), "output_tok_per_s")
                if base_thr and thr:
                    sp.append(thr / base_thr)
            return statistics.mean(sp) if sp else None
        return thr / base_thr if thr and base_thr else None
    ws.append([method, round(speed("A", 1) or 0, 2), round(speed("DS", 1) or 0, 2),
               round(speed("A", 8) or 0, 2), round(speed("DS", 8) or 0, 2)])
for cell in ws[1]:
    cell.font = Font(bold=True)
    cell.fill = HEADER_FILL
for idx in range(1, 6):
    ws.column_dimensions[get_column_letter(idx)].width = 18

wb.save(OUT)
print("saved", OUT)
