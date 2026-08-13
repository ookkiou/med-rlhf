"""
SFT 数据清洗 + 抽样

从 shibing624/medical (240 万条) 中按比例抽样 5-8 万条,
构造 sft_medical.jsonl 供 SFT 训练使用。

抽样策略:
  - 医疗问答（中文化）   50%  ~30,000
  - 医疗对话（多轮）     25%  ~15,000
  - 医学教材/百科 QA     15%  ~9,000
  - 安全合规（拒答/引导就医）10% ~6,000

用法:
    python data/prepare_medical_data.py --total 60000 --out data/sft_medical.jsonl
"""
import argparse
import json
import random
from collections import defaultdict

from huggingface_hub import hf_hub_download

# 类别关键词
SAFETY_KW = ["自杀", "自残", "过量", "处方", "用药", "剂量", "诊断自己", "怎么办", "治疗建议"]
ENCY_KW = ["什么是", "定义", "概述", "分类", "病因", "发病机制", "百科", "教材", "解剖", "生理"]


def classify(item):
    """按关键词粗分类"""
    text = (item.get("instruction") or "") + (item.get("input") or "")
    if any(k in text for k in SAFETY_KW):
        return "safety"
    if any(k in text for k in ENCY_KW):
        return "encyclopedia"
    if len(item.get("input") or "") > 100:
        return "dialog"
    return "qa"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=60000, help="抽样总数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/sft_medical.jsonl")
    parser.add_argument("--sample-check", type=int, default=50, help="抽检条数")
    args = parser.parse_args()

    random.seed(args.seed)

    # 下载中文训练集
    print("下载 shibing624/medical finetune/train_zh_0.json ...")
    local = hf_hub_download(
        repo_id="shibing624/medical",
        filename="finetune/train_zh_0.json",
        repo_type="dataset",
    )
    with open(local, encoding="utf-8") as f:
        data = json.load(f)
    print(f"原始数据: {len(data)} 条")

    # 过滤空数据
    data = [x for x in data if (x.get("instruction") or x.get("input")) and x.get("output")]
    print(f"过滤后: {len(data)} 条")

    # 按类别分组
    groups = defaultdict(list)
    for item in data:
        groups[classify(item)].append(item)
    for cat, pool in groups.items():
        print(f"  类别 {cat}: {len(pool)} 条")

    # 按比例抽样
    ratios = {"qa": 0.5, "dialog": 0.25, "encyclopedia": 0.15, "safety": 0.1}
    sampled = []
    for cat, ratio in ratios.items():
        pool = groups[cat]
        n = int(args.total * ratio)
        n = min(n, len(pool))
        picked = random.sample(pool, n)
        sampled.extend(picked)
        print(f"  {cat}: 池 {len(pool)}, 抽 {n}")

    random.shuffle(sampled)
    print(f"抽样总数: {len(sampled)}")

    # 输出
    with open(args.out, "w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已保存: {args.out}")

    # 抽检
    if args.sample_check > 0:
        check_path = args.out.replace(".jsonl", "_check.jsonl")
        with open(check_path, "w", encoding="utf-8") as f:
            for item in sampled[: args.sample_check]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"抽检样本(前 {args.sample_check} 条): {check_path}")


if __name__ == "__main__":
    main()
