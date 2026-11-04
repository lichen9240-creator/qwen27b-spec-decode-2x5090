#!/usr/bin/env python3
"""下载请求集所需的 4 个公开数据集到 data/raw/，供 build_requests.py 使用。

| 数据集   | 来源                                        | 许可证      | 落盘文件                     |
|----------|---------------------------------------------|-------------|------------------------------|
| HumanEval| openai/human-eval (GitHub raw)               | MIT         | humaneval.jsonl.gz           |
| GSM8K    | openai/grade-school-math test.jsonl          | MIT         | gsm8k_test.jsonl             |
| QASPER   | allenai/qasper，validation split             | CC BY 4.0   | qasper_validation.parquet    |
| CMRC2018 | hfl/cmrc2018，validation split               | Apache-2.0  | cmrc2018_validation.parquet  |

国内网络可先设置镜像：
    export HF_ENDPOINT=https://hf-mirror.com

用法：
    python scripts/fetch_datasets.py                 # 全部下载并校验
    python scripts/fetch_datasets.py --only humaneval,gsm8k
    python scripts/fetch_datasets.py --force         # 已存在也重新下载
"""
import argparse
import gzip
import json
import os
import sys
import urllib.request

ROOT = os.environ.get("BENCH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.environ.get("RAW_DATA_DIR", os.path.join(ROOT, "data", "raw"))
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
UA = {"User-Agent": "spec-decode-repro/1.0"}

GITHUB = {
    "humaneval": ("https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
                  "humaneval.jsonl.gz"),
    "gsm8k": ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
              "grade_school_math/data/test.jsonl", "gsm8k_test.jsonl"),
}
HF_DATASETS = {
    "qasper": {"repo": "allenai/qasper", "config": "qasper", "split": "validation",
               "file": "qasper_validation.parquet"},
    "cmrc2018": {"repo": "hfl/cmrc2018", "config": "default", "split": "validation",
                 "file": "cmrc2018_validation.parquet"},
}


def download(url, path, attempts=4):
    """带断点续传的下载：不完整就按 Range 续传，最多 attempts 次。

    部分网络的 HTTP 客户端会在末尾截断大文件，续传能把缺口补齐；
    文件非空但比 Content-Length 短时，第 2 次起改用 Range 请求。
    """
    tmp = path + ".part"
    for attempt in range(1, attempts + 1):
        done = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
        headers = dict(UA)
        if done:
            headers["Range"] = "bytes=%d-" % done
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as resp:
                resumed = resp.status == 206 and done > 0
                if not resumed:
                    done = 0
                total = int(resp.headers.get("Content-Length") or 0) + (done if resumed else 0)
                with open(tmp, "ab" if resumed else "wb") as out:
                    while True:
                        block = resp.read(1 << 20)
                        if not block:
                            break
                        out.write(block)
                        done += len(block)
                        if total:
                            sys.stdout.write("\r  %5.1f%%  %d/%d MB" % (
                                100.0 * done / total, done >> 20, total >> 20))
                            sys.stdout.flush()
                sys.stdout.write("\n")
        except Exception as exc:                      # 网络抖动：留着 .part 下次续传
            print("\n  第 %d 次下载中断：%s" % (attempt, exc))
            continue
        size = os.path.getsize(tmp)
        if not total or size >= total:
            os.replace(tmp, path)
            return path
        print("  不完整：%d/%d 字节，将从断点续传" % (size, total))
    raise IOError("下载失败（%d 次尝试后仍不完整）：%s" % (attempts, path))


def hf_parquet_url(repo, config, split):
    """查 HF 的 parquet 转换接口，拿到该 split 的第一个 parquet 直链。"""
    api = "%s/api/datasets/%s/parquet" % (HF, repo)
    with urllib.request.urlopen(urllib.request.Request(api, headers=UA), timeout=60) as resp:
        listing = json.load(resp)
    try:
        return listing[config][split][0]
    except KeyError:
        raise KeyError("在 %s 的 parquet 列表里找不到 %s/%s：%s"
                       % (repo, config, split, list(listing)))


def describe(path):
    """粗略校验：打印行数，避免下到 HTML 错误页也当成功。"""
    name = os.path.basename(path)
    size = os.path.getsize(path)
    if name.endswith(".parquet"):
        import pyarrow.parquet as pq
        meta = pq.read_metadata(path)
        return "%d 行 × %d 列, %.1f MB" % (meta.num_rows, meta.num_columns, size / 1e6)
    if name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return "%d 行, %.1f MB" % (sum(1 for _ in fh), size / 1e6)
    with open(path, encoding="utf-8") as fh:
        return "%d 行, %.1f MB" % (sum(1 for _ in fh), size / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔：humaneval,gsm8k,qasper,cmrc2018")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = ap.parse_args()
    want = [x.strip() for x in args.only.split(",") if x.strip()] or \
           list(GITHUB) + list(HF_DATASETS)

    os.makedirs(RAW, exist_ok=True)
    failed = []
    for name in want:
        if name in GITHUB:
            url, filename = GITHUB[name]
        elif name in HF_DATASETS:
            spec = HF_DATASETS[name]
            url = hf_parquet_url(spec["repo"], spec["config"], spec["split"])
            filename = spec["file"]
        else:
            print("跳过未知数据集", name)
            continue
        path = os.path.join(RAW, filename)
        if os.path.isfile(path) and not args.force:
            print("[skip] %s 已存在" % filename)
        else:
            print("[get ] %s ← %s" % (filename, url))
            try:
                download(url, path)
            except Exception as exc:                      # 下载失败不影响其它数据集
                failed.append("%s: %s" % (name, exc))
                print("  失败：%s" % exc)
                continue
        try:
            print("        %s" % describe(path))
        except Exception as exc:
            failed.append("%s: 校验失败 %s" % (name, exc))
            print("  校验失败：%s" % exc)

    print("\n数据目录：%s" % RAW)
    if failed:
        print("失败项：\n  " + "\n  ".join(failed))
        return 1
    print("全部就绪，下一步：python scripts/build_requests.py --n 25")
    return 0


if __name__ == "__main__":
    sys.exit(main())
