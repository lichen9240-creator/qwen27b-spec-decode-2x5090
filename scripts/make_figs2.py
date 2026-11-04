#!/usr/bin/env python3
"""硬件与任务类型两张图。"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("BENCH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
rows = list(csv.DictReader(open(os.environ.get("RUNS_CSV", os.path.join(RESULTS, "real_runs.csv")))))


def phase(r):
    parts = r["run_id"].split("_")
    return parts[-2] if len(parts) >= 2 else "?"


hw = {}
for r in rows:
    if phase(r) == "HW" and r["method"] not in hw:
        hw[r["method"]] = r
bot = [r for r in rows if phase(r) == "BOT"]

order = ["BASE", "MTP3", "DFLASH2", "DSPARK", "SUFFIX"]
names = [m for m in order if m in hw]
thr = [float(hw[m]["output_tok_per_s"]) for m in names]
tpw = [float(hw[m]["tokens_per_watt"] or 0) for m in names]
sm = [(float(hw[m]["gpu0_util"]) + float(hw[m]["gpu1_util"])) / 2 for m in names]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
axes[0].bar(names, thr, color="#1f77b4", alpha=0.85)
axes[0].set_ylabel("Throughput (tok/s)")
axes[0].set_title("C=4 throughput (100 real requests)")
for i, value in enumerate(thr):
    axes[0].text(i, value + 6, "%.0f" % value, ha="center", fontsize=8)

axes[1].bar(names, tpw, color="#2ca02c", alpha=0.85)
axes[1].set_ylabel("Tokens per Watt (tok/J)")
axes[1].set_title("Energy efficiency (both GPUs)")
for i, value in enumerate(tpw):
    axes[1].text(i, value + 0.015, "%.3f" % value, ha="center", fontsize=8)
for axis in axes:
    axis.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "fig_hw_efficiency.png"), dpi=150)
plt.close()

if bot:
    labels = [r["dataset"] for r in bot]
    accs = [float(r["accept_rate"] or 0) for r in bot]
    thr2 = [float(r["output_tok_per_s"]) for r in bot]
    fig, axis = plt.subplots(figsize=(8, 4.2))
    bars = axis.bar(labels, accs, color="#d62728", alpha=0.85)
    axis.set_ylabel("Acceptance rate")
    axis.set_title("DFlash2 acceptance by task type (C=1)")
    axis.set_ylim(0, 1.0)
    for bar, value, t in zip(bars, accs, thr2):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                  "%.2f\n%.0f tok/s" % (value, t), ha="center", fontsize=8)
    axis.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig_accept_by_task.png"), dpi=150)
    plt.close()
print("saved figures", os.listdir(FIGURES)[-3:])
