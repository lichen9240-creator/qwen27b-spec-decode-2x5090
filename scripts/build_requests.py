#!/usr/bin/env python3
"""构建统一请求集：HumanEval / GSM8K / QASPER / CMRC2018，去重后按固定 seed 抽样。

    python scripts/build_requests.py --n 25   # 混合 100 条（round 2 用）
    python scripts/build_requests.py --n 64   # 每类 64 条 = 256 条（round 3 按类别矩阵用）

抽样顺序只由 seed 与源数据决定，因此 N=64 的前 25 条/类与 N=25 的结果逐条一致。
"""
import argparse
import gzip
import hashlib
import json
import os
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("BENCH_ROOT", REPO)
DATA = os.environ.get("RAW_DATA_DIR", os.path.join(REPO, "data", "raw"))
OUT = os.environ.get("REQUESTS_OUT", os.path.join(REPO, "data", "requests.jsonl"))
MANIFEST = os.environ.get("MANIFEST_OUT", os.path.join(REPO, "data", "dataset_manifest.md"))
N = 25
SEED = 20261030
random.seed(SEED)


def norm(text):
    return " ".join(text.split()).strip().lower()


def dedup(items, key):
    seen, out = set(), []
    for item in items:
        sig = hashlib.sha256(norm(key(item)).encode()).hexdigest()[:16]
        if sig in seen:
            continue
        seen.add(sig)
        out.append(item)
    return out


def load_humaneval():
    rows = []
    with gzip.open(os.path.join(DATA, "humaneval.jsonl.gz"), "rt") as handle:
        for line in handle:
            item = json.loads(line)
            rows.append({
                "source_id": item["task_id"],
                "prompt": item["prompt"],
                "target": item.get("canonical_solution", "")[:1200],
            })
    return rows


def load_gsm8k():
    rows = []
    with open(os.path.join(DATA, "gsm8k_test.jsonl")) as handle:
        for index, line in enumerate(handle):
            item = json.loads(line)
            answer = item["answer"].split("####")[-1].strip()
            rows.append({
                "source_id": "gsm8k/%d" % index,
                "prompt": item["question"],
                "target": answer,
            })
    return rows


def load_qasper():
    import pandas as pd
    frame = pd.read_parquet(os.path.join(DATA, "qasper_validation.parquet"))
    rows = []
    for index, row in frame.iterrows():
        qas = row.get("qas") or {}
        raw_questions = qas.get("question")
        raw_answers = qas.get("answers")
        questions = list(raw_questions) if raw_questions is not None else []
        answers_all = list(raw_answers) if raw_answers is not None else []
        full = row.get("full_text") or {}
        paragraphs = full.get("paragraphs")
        context = " ".join(str(p) for p in list(paragraphs)[:40]) if paragraphs is not None else ""
        if not context.strip():
            context = str(row.get("abstract") or "")
        context = " ".join(context.split())
        if not context:
            continue
        for q_index, question in enumerate(questions[:2]):
            if q_index >= len(answers_all):
                continue
            answer_texts = []
            raw_list = answers_all[q_index].get("answer")
            for answer in (list(raw_list) if raw_list is not None else []):
                if isinstance(answer, dict) and answer.get("free_form_answer"):
                    answer_texts.append(answer["free_form_answer"])
            if not answer_texts:
                continue
            rows.append({
                "source_id": "qasper/%s_%d" % (str(row.get("id"))[:12], q_index),
                "prompt": "论文内容：%s\n\n问题：%s" % (context[:12000], question),
                "target": answer_texts[0][:400],
            })
    return rows


def load_cmrc():
    import pandas as pd
    path = os.path.join(DATA, "cmrc2018_validation.parquet")
    frame = pd.read_parquet(path)
    rows = []
    for index, row in frame.iterrows():
        context = row.get("context")
        question = row.get("question")
        answers = row.get("answers")
        try:
            target = answers["text"][0]
        except (TypeError, KeyError, IndexError):
            continue
        if not context or not question:
            continue
        rows.append({
            "source_id": "cmrc/%d" % index,
            "prompt": "阅读材料：%s\n\n问题：%s" % (str(context)[:6000], question),
            "target": str(target)[:200],
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N, help="每个数据集抽多少条（25=混合100，64=分类别256）")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--manifest", default=MANIFEST)
    args = parser.parse_args()
    n = args.n

    specs = [
        ("humaneval", "代码生成", load_humaneval()),
        ("gsm8k", "数学推理", load_gsm8k()),
        ("qasper", "长文档问答", load_qasper()),
        ("cmrc2018", "中文问答(客服替代)", load_cmrc()),
    ]
    total = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        for dataset, category, rows in specs:
            rows = dedup(rows, key=lambda item: item["prompt"])
            random.shuffle(rows)
            picked = rows[:n]
            for order, item in enumerate(picked):
                record = {
                    "request_id": "%s_%03d" % (dataset, order),
                    "dataset": dataset,
                    "category": category,
                    "source_id": item["source_id"],
                    "prompt": item["prompt"],
                    "target": item["target"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
            print("%-10s 候选 %4d → 选用 %d" % (dataset, len(rows), len(picked)))
    with open(args.manifest, "w", encoding="utf-8") as handle:
        handle.write("# 请求集清单\n\n| 数据集 | 类别 | 来源 | 许可证 | 条数 |\n|---|---|---|---|---|\n")
        handle.write("| HumanEval | 代码生成 | openai/human-eval (GitHub raw) | MIT | %d |\n" % n)
        handle.write("| GSM8K | 数学推理 | openai/grade-school-math test.jsonl | MIT | %d |\n" % n)
        handle.write("| QASPER | 长文档问答 | allenai/qasper validation parquet | CC BY 4.0 | %d |\n" % n)
        handle.write("| CMRC2018 | 中文问答（JDDC 替代） | hfl/cmrc2018 validation parquet | Apache-2.0 | %d |\n" % n)
        handle.write("\n> JDDC 2.1 需申请，且 HF 上四个镜像均返回 401；按任务书改用可获取的中文问答集 CMRC2018，"
                     "类别标注为「中文问答(客服替代)」。\n")
        handle.write("\n抽样规则：先按 prompt 归一化去重，再随机抽样（seed=%d），每类 %d 条。\n" % (SEED, n))
    print("total", total, "->", args.out)
    return 0


if __name__ == "__main__":
    main()
