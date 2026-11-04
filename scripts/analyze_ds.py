#!/usr/bin/env python3
"""按类别 × 方法 × 并发聚合（phase DS），输出中位数/标准差/加速比与排名。"""
import csv
import os
import statistics
from collections import defaultdict

ROOT = os.environ.get("BENCH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.environ.get("RUNS_CSV", os.path.join(ROOT, "results", "real_runs.csv"))
OUT = os.environ.get("DS_SUMMARY_OUT", os.path.join(ROOT, "results", "per_dataset_summary.md"))
DATASETS = ["humaneval", "gsm8k", "qasper", "cmrc2018"]
LABEL = {"humaneval": "代码", "gsm8k": "数学", "qasper": "长文档", "cmrc2018": "中文问答"}


def phase(row):
    parts = row["run_id"].split("_")
    for token in (parts[-1], parts[-2] if len(parts) >= 2 else ""):
        if token in ("A", "HW", "LC", "Q", "BOT", "DS", "PROBE"):
            return token
    return "?"


rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if phase(r) == "DS"]
print("DS 行数:", len(rows))

buckets = defaultdict(list)
for row in rows:
    try:
        conc = int(row["concurrency"])
    except (TypeError, ValueError):
        continue
    buckets[(row["method"], row["dataset"], conc)].append(row)

stats = {}
for key, items in buckets.items():
    def col(name, cast=float):
        vals = []
        for item in items:
            try:
                if item.get(name):
                    vals.append(cast(item[name]))
            except ValueError:
                pass
        return vals
    thr = col("output_tok_per_s")
    stats[key] = {
        "n": len(thr),
        "thr": statistics.median(thr) if thr else None,
        "thr_sd": statistics.pstdev(thr) if len(thr) > 1 else 0.0,
        "ttft": statistics.median(col("ttft_p50") or [0]) if col("ttft_p50") else None,
        "tpot": statistics.median(col("tpot_p50") or [0]) if col("tpot_p50") else None,
        "acc": statistics.median(col("accept_rate")) if col("accept_rate") else None,
        "kv": statistics.median(col("kv_tokens")) if col("kv_tokens") else None,
    }

lines = ["# 按类别结果（64 条/类，3 次重复）\n"]
for dataset in DATASETS:
    lines.append("\n## %s（%s，64 条）\n" % (dataset, LABEL[dataset]))
    lines.append("| 方法 | C=1 | C=4 | C=8 | C=16 | C=32 | 加速比 C=1 / C=8 | 接受率 C=1 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    base = {c: stats.get(("BASE", dataset, c), {}).get("thr") for c in (1, 4, 8, 16, 32)}
    for method in ["BASE", "MTP3", "DFLASH2", "DSPARK", "SUFFIX"]:
        cells = []
        for conc in (1, 4, 8, 16, 32):
            value = stats.get((method, dataset, conc), {}).get("thr")
            cells.append("%.0f" % value if value else "—")
        speed1 = speed8 = None
        if base[1] and stats.get((method, dataset, 1), {}).get("thr"):
            speed1 = stats[(method, dataset, 1)]["thr"] / base[1]
        if base[8] and stats.get((method, dataset, 8), {}).get("thr"):
            speed8 = stats[(method, dataset, 8)]["thr"] / base[8]
        acc = stats.get((method, dataset, 1), {}).get("acc")
        lines.append("| %s | %s | %s | %s |" % (
            method, " | ".join(cells),
            ("%.2fx / %.2fx" % (speed1, speed8)) if speed1 and speed8 else "—",
            ("%.3f" % acc) if acc else "—"))

lines.append("\n## 每个类别的最优方法（按 C=1 加速比）\n")
lines.append("| 类别 | 第 1 | 第 2 | 第 3 |")
lines.append("|---|---|---|---|")
for dataset in DATASETS:
    base_thr = stats.get(("BASE", dataset, 1), {}).get("thr")
    ranking = []
    for method in ["MTP3", "DFLASH2", "DSPARK", "SUFFIX"]:
        value = stats.get((method, dataset, 1), {}).get("thr")
        if value and base_thr:
            ranking.append((value / base_thr, method))
    ranking.sort(reverse=True)
    cells = ["%s %.2fx" % (m, s) for s, m in ranking[:3]]
    lines.append("| %s（%s） | %s |" % (dataset, LABEL[dataset], " | ".join(cells)))

print("\n".join(lines))
open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\nsaved", OUT)
