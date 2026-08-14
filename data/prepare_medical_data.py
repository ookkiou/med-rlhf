"""
SFT 数据准备：三源混合

数据来源:
  1. shibing624/medical (240万条) → 医疗问答/对话/百科/安全 (~70%)
  2. FreedomIntelligence/medical-o1-reasoning-SFT (9万条) → 医疗 CoT 推理 (~20%)
  3. llamafactory/alpaca_zh (5万条) → 通用指令防遗忘 (~10%)

输出格式: JSONL, 每行 {instruction, input, output}

用法:
    python data/prepare_medical_data.py --total 50000 --out data/sft_medical.jsonl
"""
import argparse
import json
import random
from collections import defaultdict

from datasets import load_dataset
from huggingface_hub import hf_hub_download

SAFETY_KW = ["自杀", "自残", "过量", "处方", "用药", "剂量", "诊断自己", "怎么办", "治疗建议"]
ENCY_KW = ["什么是", "定义", "概述", "分类", "病因", "发病机制", "百科", "教材", "解剖", "生理"]


def classify(item):
    text = (item.get("instruction") or "") + (item.get("input") or "")
    if any(k in text for k in SAFETY_KW):
        return "safety"
    if any(k in text for k in ENCY_KW):
        return "encyclopedia"
    if len(item.get("input") or "") > 100:
        return "dialog"
    return "qa"


def load_shibing624(total_medical):
    """加载 shibing624/medical，按类别抽样"""
    print("\n[1/3] 加载 shibing624/medical ...")
    local = hf_hub_download(
        repo_id="shibing624/medical",
        filename="finetune/train_zh_0.json",
        repo_type="dataset",
    )
    with open(local, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    data = [x for x in data if (x.get("instruction") or x.get("input")) and x.get("output")]
    print(f"  过滤后: {len(data)} 条")

    groups = defaultdict(list)
    for item in data:
        groups[classify(item)].append(item)

    ratios = {"qa": 0.5, "dialog": 0.25, "encyclopedia": 0.15, "safety": 0.1}
    sampled = []
    for cat, ratio in ratios.items():
        pool = groups[cat]
        n = min(int(total_medical * ratio), len(pool))
        picked = random.sample(pool, n)
        sampled.extend(picked)
        print(f"  {cat}: 池 {len(pool)}, 抽 {n}")

    print(f"  shibing624 小计: {len(sampled)} 条")
    return sampled


def load_huatuogpt_cot(total_cot):
    """加载 HuatuoGPT-o1 医疗 CoT 推理数据"""
    print("\n[2/3] 加载 medical-o1-reasoning-SFT (CoT) ...")
    ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", split="train")
    print(f"  原始数据: {len(ds)} 条")

    items = []
    for row in ds:
        question = row.get("Question") or row.get("question") or ""
        cot = row.get("Complex_CoT") or row.get("complex_cot") or ""
        response = row.get("Response") or row.get("response") or ""
        if not question or not response:
            continue
        output = f"{cot}\n\n{response}" if cot else response
        items.append({"instruction": question, "input": "", "output": output})

    n = min(total_cot, len(items))
    sampled = random.sample(items, n)
    print(f"  CoT 小计: {len(sampled)} 条")
    return sampled


def load_general_instructions(total_general):
    """加载通用中文指令数据，防止灾难性遗忘"""
    print("\n[3/3] 加载 alpaca_zh (通用指令) ...")
    ds = load_dataset("llamafactory/alpaca_zh", split="train")
    print(f"  原始数据: {len(ds)} 条")

    items = []
    for row in ds:
        inst = row.get("instruction") or ""
        inp = row.get("input") or ""
        out = row.get("output") or ""
        if not inst or not out:
            continue
        items.append({"instruction": inst, "input": inp, "output": out})

    n = min(total_general, len(items))
    sampled = random.sample(items, n)
    print(f"  通用指令小计: {len(sampled)} 条")
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=50000, help="总抽样数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/sft_medical.jsonl")
    parser.add_argument("--sample-check", type=int, default=50, help="抽检条数")
    args = parser.parse_args()

    random.seed(args.seed)

    # 按比例分配: 70% 医疗QA + 20% CoT + 10% 通用
    total_medical = int(args.total * 0.70)
    total_cot = int(args.total * 0.20)
    total_general = int(args.total * 0.10)

    print(f"=== SFT 数据准备 (总计 {args.total} 条) ===")
    print(f"  医疗QA: {total_medical} (70%)")
    print(f"  医疗CoT: {total_cot} (20%)")
    print(f"  通用指令: {total_general} (10%)")

    medical_data = load_shibing624(total_medical)
    cot_data = load_huatuogpt_cot(total_cot)
    general_data = load_general_instructions(total_general)

    all_data = medical_data + cot_data + general_data
    random.shuffle(all_data)

    print(f"\n=== 总计: {len(all_data)} 条 ===")

    with open(args.out, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已保存: {args.out}")

    if args.sample_check > 0:
        check_path = args.out.replace(".jsonl", "_check.jsonl")
        with open(check_path, "w", encoding="utf-8") as f:
            for item in all_data[: args.sample_check]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"抽检样本(前 {args.sample_check} 条): {check_path}")


if __name__ == "__main__":
    main()
