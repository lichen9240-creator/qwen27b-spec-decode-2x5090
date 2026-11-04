#!/usr/bin/env python3
"""按类别结果的三张图：加速比热力图、分数据集并发曲线、接受率分组柱状图。"""
import csv
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.environ.get("BENCH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.environ.get("RUNS_CSV", os.path.join(ROOT, "results", "real_runs.csv"))
FIGURES = os.environ.get("FIGURES_DIR", os.path.join(ROOT, "figures"))
DATASETS = ["humaneval", "gsm8k", "qasper", "cmrc2018"]
LABEL = {"humaneval": "HumanEval (code)", "gsm8k": "GSM8K (math)",
         "qasper": "QASPER (long doc)", "cmrc2018": "CMRC (Chinese)"}
METHODS = ["MTP3", "DFLASH2", "DSPARK", "SUFFIX"]
COLORS = {"BASE": "#4C4C4C", "MTP3": "#1f77b4", "DFLASH2": "#d62728",
          "DSPARK": "#ff7f0e", "SUFFIX": "#9467bd"}
MARKERS = {"BASE": "s", "MTP3": "o", "DFLASH2": "D", "DSPARK": "^", "SUFFIX": "P"}


def phase(row):
    parts = row["run_id"].split("_")
    for token in (parts[-1], parts[-2] if len(parts) >= 2 else ""):
        if token in ("A", "HW", "LC", "Q", "BOT", "DS", "PROBE"):
            return token
    return "?"


rows = list(csv.DictReader(open(CSV)))
buckets = defaultdict(list)
for row in rows:
    if phase(row) != "DS":
        continue
    try:
        buckets[(row["dataset"], row["method"], int(row["concurrency"]))].append(row)
    except (TypeError, ValueError):
        continue


def med(dataset, method, conc, key):
    items = buckets.get((dataset, method, conc), [])
    vals = []
    for item in items:
        try:
            if item.get(key):
                vals.append(float(item[key]))
        except ValueError:
            pass
    return statistics.median(vals) if vals else None


# 1) 热力图：C=1 与 C=8 的加速比
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for axis, conc in zip(axes, (1, 8)):
    matrix = []
    for method in METHODS:
        line = []
        for dataset in DATASETS:
            base = med(dataset, "BASE", conc, "output_tok_per_s")
            value = med(dataset, method, conc, "output_tok_per_s")
            line.append(value / base if base and value else 0)
        matrix.append(line)
    im = axis.imshow(matrix, cmap="RdYlGn", vmin=0.8, vmax=3.0, aspect="auto")
    axis.set_xticks(range(len(DATASETS)))
    axis.set_xticklabels([LABEL[d].split(" ")[0] for d in DATASETS], fontsize=8)
    axis.set_yticks(range(len(METHODS)))
    axis.set_yticklabels(METHODS, fontsize=8)
    axis.set_title("Speedup vs BASE @ C=%d" % conc, fontsize=9)
    for i in range(len(METHODS)):
        for j in range(len(DATASETS)):
            axis.text(j, i, "%.2f" % matrix[i][j], ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=axes, shrink=0.8)
plt.savefig(os.path.join(FIGURES, "fig_ds_speedup_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# 2) 分数据集并发曲线
fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
for axis, dataset in zip(axes, DATASETS):
    for method in ["BASE"] + METHODS:
        xs, ys = [], []
        for conc in (1, 4, 8, 16, 32):
            value = med(dataset, method, conc, "output_tok_per_s")
            if value:
                xs.append(conc)
                ys.append(value)
        if xs:
            axis.plot(xs, ys, marker=MARKERS[method], color=COLORS[method],
                      linewidth=1.6, markersize=5, label=method)
    axis.set_xscale("log", base=2)
    axis.set_title(LABEL[dataset], fontsize=9)
    axis.set_xlabel("Concurrency", fontsize=8)
    axis.grid(alpha=0.3)
axes[0].set_ylabel("Output tok/s")
axes[0].legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "fig_ds_curves.png"), dpi=150)
plt.close()

# 3) 接受率分组柱状
fig, axis = plt.subplots(figsize=(9, 3.8))
width = 0.2
xs = np.arange(len(DATASETS))
for index, method in enumerate(METHODS):
    values = [med(dataset, method, 1, "accept_rate") or 0 for dataset in DATASETS]
    axis.bar(xs + (index - 1.5) * width, values, width, label=method,
             color=COLORS[method], alpha=0.9)
axis.set_xticks(xs)
axis.set_xticklabels([LABEL[d] for d in DATASETS], fontsize=8)
axis.set_ylabel("Acceptance rate @ C=1")
axis.set_title("Acceptance by task type")
axis.grid(alpha=0.3, axis="y")
axis.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "fig_ds_acceptance.png"), dpi=150)
plt.close()
print("saved 3 figures")
