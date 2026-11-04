#!/usr/bin/env python3
"""任务级质量评分（Track B）：中文用字符级 F1，代码先去 markdown 围栏再语法检查。"""
import ast
import glob
import json
import os
import re
import string
import sys
from collections import defaultdict

E = os.environ.get("BENCH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REQ_PATH = os.environ.get("REQUESTS_JSONL", os.path.join(E, "data", "requests.jsonl"))
REQ = {json.loads(l)["request_id"]: json.loads(l)
       for l in open(REQ_PATH)}


def is_cjk(text):
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk / max(1, len(text)) > 0.2


def norm_tokens(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t]


def f1(pred, gold):
    if is_cjk(gold) or is_cjk(pred):
        a = [ch for ch in pred if not ch.isspace()]
        b = [ch for ch in gold if not ch.isspace()]
    else:
        a, b = norm_tokens(pred), norm_tokens(gold)
    if not a or not b:
        return 0.0
    pool = list(b)
    common = 0
    for token in a:
        if token in pool:
            pool.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(a), common / len(b)
    return 2 * precision * recall / (precision + recall)


def strip_fence(text):
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return match.group(1) if match else text


def number(text):
    hits = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return hits[-1] if hits else ""


def score_one(item):
    record = REQ.get(item["request_id"])
    if not record:
        return None, "n/a"
    text = item.get("text") or ""
    gold = record.get("target") or ""
    dataset = record["dataset"]
    if dataset == "gsm8k":
        return (1.0 if number(text) and number(text) == number(gold) else 0.0), "exact_number"
    if dataset in ("cmrc2018", "qasper"):
        return f1(text, gold), "char_f1" if is_cjk(gold) else "token_f1"
    if dataset == "humaneval":
        code = strip_fence(text)
        try:
            ast.parse(code)
            return 1.0, "syntax_ok"
        except SyntaxError:
            return 0.0, "syntax_ok"
    return None, "n/a"


def main(pattern):
    total = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(E, "raw_data", pattern))):
        method = os.path.basename(path).split("_")[1]
        for line in open(path):
            item = json.loads(line)
            if not item.get("ok"):
                continue
            value, kind = score_one(item)
            dataset = REQ.get(item["request_id"], {}).get("dataset", "?")
            if value is not None:
                total[method][dataset].append((value, kind))
    print("%-10s %-10s %6s %7s %10s" % ("method", "dataset", "n", "score", "metric"))
    for method, by_dataset in sorted(total.items()):
        for dataset, pairs in sorted(by_dataset.items()):
            values = [v for v, _ in pairs]
            print("%-10s %-10s %6d %7.3f %10s" % (method, dataset, len(values),
                                                  sum(values) / len(values), pairs[0][1]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "*_Q_*.jsonl")
