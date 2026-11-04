#!/usr/bin/env python3
"""汇总 real_runs.csv：中位数表 + 加速比 + 图表。"""
import csv
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("BENCH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.environ.get("RUNS_CSV", os.path.join(ROOT, "results", "real_runs.csv"))
OUT_MD = os.path.join(ROOT, "results", "summary_table.md")
PLOTS = os.environ.get("FIGURES_DIR", os.path.join(ROOT, "figures"))

STYLE = {"BASE": ("#4C4C4C", "s"), "MTP3": ("#1f77b4", "o"), "SUFFIX": ("#9467bd", "P"),
         "DFLASH2": ("#d62728", "D"), "DSPARK": ("#ff7f0e", "^"),
         "EAGLE4N": ("#2ca02c", "v"), "EAGLE4H": ("#17becf", "*")}


def load():
    rows = []
    with open(CSV) as fh:
        for row in csv.DictReader(fh):
            if not row.get("concurrency"):
                continue
            for key in ("output_tok_per_s", "ttft_p50", "ttft_p95", "tpot_p50", "itl_p99",
                        "accept_rate", "gpu0_mem_gib", "kv_tokens", "tokens_per_watt"):
                row[key] = float(row[key]) if row.get(key) else None
            row["concurrency"] = int(row["concurrency"])
            row["rep"] = int(row["rep"])
            rows.append(row)
    return rows


def med(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def main():
    rows = load()
    os.makedirs(PLOTS, exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["concurrency"], row["dataset"])].append(row)

    lines = ["| 方法 | 并发 | 数据集 | n | 吞吐 tok/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p99 | 接受率 | 显存 GiB | tok/W |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = defaultdict(dict)
    for (method, conc, dataset), items in sorted(groups.items()):
        if dataset != "all" and dataset != "mixed100":
            continue
        stat = {k: med([i[k] for i in items]) for k in
                ("output_tok_per_s", "ttft_p50", "ttft_p95", "tpot_p50", "itl_p99",
                 "accept_rate", "gpu0_mem_gib", "tokens_per_watt")}
        summary[(method, conc)] = stat
        lines.append("| %s | %d | %s | %d | %.1f | %.0f | %.0f | %.2f | %.1f | %s | %.2f | %s |" % (
            method, conc, dataset, len(items), stat["output_tok_per_s"] or 0,
            stat["ttft_p50"] or 0, stat["ttft_p95"] or 0, stat["tpot_p50"] or 0,
            stat["itl_p99"] or 0,
            ("%.3f" % stat["accept_rate"]) if stat["accept_rate"] else "-",
            stat["gpu0_mem_gib"] or 0,
            ("%.3f" % stat["tokens_per_watt"]) if stat["tokens_per_watt"] else "-"))

    base = {c: summary[("BASE", c)]["output_tok_per_s"] for (m, c) in summary if m == "BASE"}
    lines.append("")
    lines.append("| 方法 | C=1 加速 | C=4 加速 | C=8 加速 |")
    lines.append("|---|---|---|---|")
    for method in sorted({m for (m, c) in summary}):
        cells = []
        for conc in (1, 4, 8):
            value = summary.get((method, conc), {}).get("output_tok_per_s")
            base_value = base.get(conc)
            cells.append("%.2fx" % (value / base_value) if value and base_value else "-")
        lines.append("| %s | %s |" % (method, " | ".join(cells)))

    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[-8:]))
    print("saved", OUT_MD)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for method in sorted({m for (m, c) in summary}):
        concs = sorted(c for (m, c) in summary if m == method)
        if not concs:
            continue
        color, marker = STYLE.get(method, ("#333333", "x"))
        axes[0].plot(concs, [summary[(method, c)]["output_tok_per_s"] for c in concs],
                     marker=marker, color=color, linewidth=1.8, label=method)
        axes[1].plot(concs, [summary[(method, c)]["tpot_p50"] for c in concs],
                     marker=marker, color=color, linewidth=1.8, label=method)
    for axis, label in zip(axes, ("Output throughput (tok/s)", "TPOT p50 (ms)")):
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Concurrency")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_title("Real requests (100 mixed): throughput")
    axes[1].set_title("Real requests: TPOT")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "fig_real_concurrency.png"), dpi=150)
    print("saved fig_real_concurrency.png")


if __name__ == "__main__":
    main()
