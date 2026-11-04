#!/usr/bin/env python3
"""从 results/real_runs.csv 重算报告里的全部表格，并和 README 中的关键数字对照。

不需要 GPU、不需要模型、不联网，只要 Python 3.8+ 标准库：

    python scripts/reproduce_tables.py            # 打印表格 + 一致性检查
    python scripts/reproduce_tables.py --check    # 只做检查，失败返回非 0
    python scripts/reproduce_tables.py --out results/recomputed-tables.md

检查口径与报告一致：
    phase=A   混合 100 条真实请求，C=1/4/8，每格 3 次重复 → 取中位数
    phase=DS  4 数据集各 64 条，C=1/4/8/16/32，每格 3 次重复 → 取中位数
"""
import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.environ.get("BENCH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.environ.get("RUNS_CSV", os.path.join(ROOT, "results", "real_runs.csv"))

METHOD_ORDER = ["BASE", "MTP3", "DFLASH2", "DSPARK", "SUFFIX"]
DATASETS = ["humaneval", "gsm8k", "qasper", "cmrc2018"]
LABEL = {"humaneval": "代码 HumanEval", "gsm8k": "数学 GSM8K",
         "qasper": "长文档 QASPER", "cmrc2018": "中文问答 CMRC2018"}

# README 里对外引用的头条数字；改代码或换数据后这里必须仍然成立（1% 容差）
EXPECT = {
    ("mixed100", "DFLASH2", 1): 203.5,
    ("mixed100", "BASE", 1): 72.9,
    ("mixed100", "MTP3", 8): 666.3,
    ("mixed100", "SUFFIX", 8): 839.7,
    ("humaneval", "DFLASH2", 1): 320.0,
    ("humaneval", "BASE", 1): 77.0,
    ("gsm8k", "DFLASH2", 1): 262.0,
    ("qasper", "BASE", 16): 694.0,
    ("qasper", "BASE", 32): 356.0,
    ("cmrc2018", "DFLASH2", 1): 152.0,
}


def phase_of(run_id):
    """run_id 形如 20261031_BASE_C16_R01_DS_qasper / 20261030_BASE_C01_R01_A。"""
    parts = run_id.split("_")
    for token in (parts[-2] if len(parts) >= 2 else "", parts[-1]):
        if token in ("A", "HW", "LC", "Q", "BOT", "DS", "KV", "BNB", "PROBE"):
            return token
    return "?"


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["phase"] = phase_of(row["run_id"])
            try:
                row["concurrency"] = int(row["concurrency"])
            except (TypeError, ValueError):
                continue
            rows.append(row)
    return rows


def num(row, key):
    value = row.get(key)
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def medians(rows, phase):
    """(数据集, 方法, 并发) → 各指标中位数。"""
    groups = defaultdict(list)
    for row in rows:
        if row["phase"] != phase:
            continue
        groups[(row["dataset"], row["method"], row["concurrency"])].append(row)
    out = {}
    for key, items in groups.items():
        stat = {}
        for field in ("output_tok_per_s", "ttft_p50", "tpot_p50", "accept_rate",
                      "tokens_per_watt", "gpu0_util", "gpu1_util"):
            vals = [v for v in (num(r, field) for r in items) if v is not None]
            stat[field] = statistics.median(vals) if vals else None
        stat["n"] = len(items)
        out[key] = stat
    return out


def fmt(value, spec="%.1f"):
    return "-" if value is None else spec % value


def table_mixed(stats, lines):
    lines.append("### 混合 100 条真实请求（phase A，C=1/4/8，3 次重复中位数）\n")
    lines.append("| 方法 | C=1 | C=4 | C=8 | 加速 1 / 4 / 8 | TPOT C=1 (ms) | 接受率 C=1 |")
    lines.append("|---|---|---|---|---|---|---|")
    base = {c: (stats.get(("mixed100", "BASE", c)) or {}).get("output_tok_per_s") for c in (1, 4, 8)}
    for method in METHOD_ORDER:
        cells, speed = [], []
        for conc in (1, 4, 8):
            value = (stats.get(("mixed100", method, conc)) or {}).get("output_tok_per_s")
            cells.append(fmt(value))
            speed.append("%.2fx" % (value / base[conc]) if value and base.get(conc) else "-")
        tpot = (stats.get(("mixed100", method, 1)) or {}).get("tpot_p50")
        acc = (stats.get(("mixed100", method, 1)) or {}).get("accept_rate")
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            method, cells[0], cells[1], cells[2], " / ".join(speed),
            fmt(tpot, "%.2f"), fmt(acc, "%.3f")))
    lines.append("")


def table_dataset(stats, lines):
    for dataset in DATASETS:
        lines.append("### %s（phase DS，64 条/次，3 次重复中位数）\n" % LABEL[dataset])
        lines.append("| 方法 | C=1 | C=4 | C=8 | C=16 | C=32 | 加速 C=1 / C=8 / C=32 | 接受率 C=1 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        base = {c: (stats.get((dataset, "BASE", c)) or {}).get("output_tok_per_s")
                for c in (1, 4, 8, 16, 32)}
        for method in METHOD_ORDER:
            cells, speed = [], []
            for conc in (1, 4, 8, 16, 32):
                value = (stats.get((dataset, method, conc)) or {}).get("output_tok_per_s")
                cells.append(fmt(value, "%.0f") if value else "-")
            for conc in (1, 8, 32):
                value = (stats.get((dataset, method, conc)) or {}).get("output_tok_per_s")
                base_value = base.get(conc)
                speed.append("%.2fx" % (value / base_value) if value and base_value else "-")
            acc = (stats.get((dataset, method, 1)) or {}).get("accept_rate")
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                method, cells[0], cells[1], cells[2], cells[3], cells[4],
                " / ".join(speed), fmt(acc, "%.3f")))
        lines.append("")


def table_hardware(rows, lines):
    lines.append("### 硬件与能效（phase HW，C=4，双卡）\n")
    lines.append("| 方法 | 吞吐 tok/s | SM 利用率 卡0 / 卡1 | 功耗 W 卡0 / 卡1 | 每瓦 token |")
    lines.append("|---|---|---|---|---|")
    seen = {}
    for row in rows:
        if row["phase"] == "HW" and row["method"] not in seen:
            seen[row["method"]] = row
    for method in METHOD_ORDER:
        row = seen.get(method)
        if not row:
            continue
        lines.append("| %s | %s | %s / %s | %s / %s | %s |" % (
            method, fmt(num(row, "output_tok_per_s")), fmt(num(row, "gpu0_util")),
            fmt(num(row, "gpu1_util")), fmt(num(row, "gpu0_power")), fmt(num(row, "gpu1_power")),
            fmt(num(row, "tokens_per_watt"), "%.3f")))
    lines.append("")


def table_probe(rows, lines):
    lines.append("### 按任务类型的接受率探测（phase BOT，DFLASH2，C=1，每类 25 条单次）\n")
    lines.append("| 数据集 | 吞吐 tok/s | TTFT p50 (ms) | TPOT p50 (ms) | 接受率 |")
    lines.append("|---|---|---|---|---|")
    for row in sorted((r for r in rows if r["phase"] == "BOT"), key=lambda r: r["dataset"]):
        lines.append("| %s | %s | %s | %s | %s |" % (
            row["dataset"], fmt(num(row, "output_tok_per_s")), fmt(num(row, "ttft_p50")),
            fmt(num(row, "tpot_p50")), fmt(num(row, "accept_rate"), "%.3f")))
    lines.append("")


def check(rows, stats):
    """把重算值和 README 引用值对照；返回 (passed, failures)。"""
    findings, failures = [], []
    expected_rows = {"A": 45, "DS": 300}
    for phase, count in expected_rows.items():
        actual = sum(1 for r in rows if r["phase"] == phase)
        findings.append("%-4s 行数 %d（期望 %d）%s" % (
            phase, actual, count, "OK" if actual == count else "不符"))
        if actual != count:
            failures.append("phase %s 行数 %d != %d" % (phase, actual, count))

    for (group, method, conc), want in sorted(EXPECT.items(), key=lambda kv: str(kv[0])):
        phase = "A" if group == "mixed100" else "DS"
        dataset = "mixed100" if group == "mixed100" else group
        got = (stats.get((dataset, method, conc)) or {}).get("output_tok_per_s")
        if got is None:
            failures.append("%s/%s/C=%d 缺失" % (group, method, conc))
            findings.append("%-9s %-7s C=%-2d 缺失" % (group, method, conc))
            continue
        delta = abs(got - want) / want
        ok = delta <= 0.01
        findings.append("%-9s %-7s C=%-2d 重算 %7.1f / 报告 %7.1f  偏差 %.2f%% %s" % (
            group, method, conc, got, want, delta * 100, "OK" if ok else "超差"))
        if not ok:
            failures.append("%s/%s/C=%d %.1f vs %.1f" % (group, method, conc, got, want))
    return findings, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="只做一致性检查")
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "recomputed-tables.md"))
    args = parser.parse_args()

    if not os.path.isfile(CSV_PATH):
        print("找不到 %s" % CSV_PATH)
        return 2
    rows = load(CSV_PATH)
    if not rows:
        print("%s 里没有数据行" % CSV_PATH)
        return 2
    stats = medians(rows, "A")
    stats.update(medians(rows, "DS"))

    lines = ["# 从 real_runs.csv 重算的结果\n",
             "> 由 `scripts/reproduce_tables.py` 生成，不需要 GPU。",
             "> 数据：`results/real_runs.csv`（%d 次运行）。\n" % len(rows)]
    table_mixed(stats, lines)
    table_dataset(stats, lines)
    table_hardware(rows, lines)
    table_probe(rows, lines)

    findings, failures = check(rows, stats)
    lines.append("## 一致性检查\n")
    lines.append("```")
    lines.extend(findings)
    lines.append("```")
    lines.append("")
    lines.append("结论：%s" % ("全部通过" if not failures else "有 %d 项不符" % len(failures)))

    text = "\n".join(lines)
    if not args.check:
        print(text)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")

    print("\n一致性检查：%d 项，%d 项失败" % (len(findings), len(failures)))
    for item in failures:
        print("  ✗", item)
    if not args.check:
        print("已写入", args.out)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
