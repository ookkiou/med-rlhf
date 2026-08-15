#!/usr/bin/env python3
"""
SFT 数据准备 v3：cMedQA2 医生回答骨干 + CoT 推理 + 通用指令防遗忘

核心改进 vs v2:
  - 以真实医生回答为主（cMedQA2），而非模板化医疗 QA（shibing624/medical）
  - 严格排除评测集 qid（test_candidates + eval_set），避免数据泄漏
  - 控制规模 ~8k，快速验证方向是否正确

数据来源:
  1. cMedQA2 (本地 CSV) → 真实医生问答骨干 (~60%)
  2. HuatuoGPT-o1 CoT → 医疗推理链 (~25%)
  3. alpaca_zh → 通用指令防遗忘 (~15%)

用法:
  python data/prepare_medical_data_v3.py --total 8000 --out data/sft_medical_v3.jsonl
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cmedqa2"


def read_csv_flexible(path):
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法解析文件编码: {path}")


def load_exclude_qids():
    """加载需要排除的 qid：test_candidates + eval_set，防止评测集泄漏"""
    exclude = set()
    c_txt = DATA_DIR / "test_candidates.txt"
    if c_txt.exists():
        with open(c_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("question_id"):
                    continue
                parts = line.replace(",", " ").split()
                if parts:
                    exclude.add(parts[0])
        print(f"  test_candidates: {len(exclude)} 个 qid")

    eval_path = DATA_DIR / "eval_set.jsonl"
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    qid = item.get("qid", "")
                    if qid:
                        exclude.add(qid)
        print(f"  加上 eval_set: {len(exclude)} 个 qid (去重后)")

    return exclude


def load_cmedqa2_doctor_answers(n_samples, exclude_qids, seed):
    """加载 cMedQA2 真实医生回答，排除测试集 qid"""
    print("\n[1/3] 加载 cMedQA2 医生回答 ...")

    q_csv = next(DATA_DIR.rglob("question*.csv"), None)
    a_csv = next(DATA_DIR.rglob("answer*.csv"), None)
    if not q_csv or not a_csv:
        print("  ✗ 未找到 question.csv / answer.csv，请先解压数据")
        sys.exit(1)

    questions = read_csv_flexible(q_csv)
    answers = read_csv_flexible(a_csv)
    print(f"  问题: {len(questions)} 条, 回答: {len(answers)} 条")

    # 按 question_id 聚合回答，取最长者（通常更完整）
    ref_by_qid = {}
    for row in answers:
        qid = (row.get("question_id") or "").strip()
        content = (row.get("content") or "").strip()
        if qid and content:
            if qid not in ref_by_qid or len(content) > len(ref_by_qid[qid]):
                ref_by_qid[qid] = content
    print(f"  有回答的问题: {len(ref_by_qid)} 个")

    # 组装训练候选：排除测试集，只保留有回答的
    pool = []
    for row in questions:
        qid = (row.get("question_id") or "").strip()
        content = (row.get("content") or "").strip()
        if qid in exclude_qids:
            continue
        ref = ref_by_qid.get(qid, "")
        if len(content) >= 5 and len(ref) >= 20:
            pool.append({
                "instruction": content,
                "input": "",
                "output": ref,
                "source": "cmedqa2",
            })

    print(f"  可用训练候选: {len(pool)} 条")
    n = min(n_samples, len(pool))
    random.seed(seed)
    sampled = random.sample(pool, n)
    print(f"  抽样: {n} 条")
    return sampled


def load_huatuogpt_cot(n_samples, seed):
    """加载 HuatuoGPT-o1 医疗 CoT 推理数据"""
    print("\n[2/3] 加载 medical-o1-reasoning-SFT (CoT) ...")
    ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "zh", split="train")
    items = []
    for row in ds:
        question = row.get("Question") or row.get("question") or ""
        cot = row.get("Complex_CoT") or row.get("complex_cot") or ""
        response = row.get("Response") or row.get("response") or ""
        if not question or not response:
            continue
        output = f"{cot}\n\n{response}" if cot else response
        items.append({"instruction": question, "input": "", "output": output, "source": "cot"})
    print(f"  原始: {len(ds)}, 过滤后: {len(items)}")
    n = min(n_samples, len(items))
    random.seed(seed)
    sampled = random.sample(items, n)
    print(f"  抽样: {n} 条")
    return sampled


def load_general_instructions(n_samples, seed):
    """加载通用中文指令数据，防止灾难性遗忘"""
    print("\n[3/3] 加载 alpaca_zh (通用指令) ...")
    ds = load_dataset("llamafactory/alpaca_zh", split="train")
    items = []
    for row in ds:
        inst = row.get("instruction") or ""
        inp = row.get("input") or ""
        out = row.get("output") or ""
        if not inst or not out:
            continue
        items.append({"instruction": inst, "input": inp, "output": out, "source": "general"})
    print(f"  原始: {len(ds)}, 过滤后: {len(items)}")
    n = min(n_samples, len(items))
    random.seed(seed)
    sampled = random.sample(items, n)
    print(f"  抽样: {n} 条")
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=8000, help="总抽样数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/sft_medical_v3.jsonl")
    args = parser.parse_args()

    print(f"=== SFT 数据准备 v3 (总计 {args.total} 条) ===\n")

    # 加载排除 qid
    print("加载排除 qid（防止泄漏）...")
    exclude_qids = load_exclude_qids()

    # 按比例分配: 60% cMedQA2 + 25% CoT + 15% 通用
    n_cmedqa = int(args.total * 0.60)
    n_cot = int(args.total * 0.25)
    n_general = args.total - n_cmedqa - n_cot

    print(f"  分配: cMedQA2={n_cmedqa} CoT={n_cot} 通用={n_general}")

    data = []
    data += load_cmedqa2_doctor_answers(n_cmedqa, exclude_qids, args.seed)
    data += load_huatuogpt_cot(n_cot, args.seed + 1)
    data += load_general_instructions(n_general, args.seed + 2)

    random.seed(args.seed)
    random.shuffle(data)

    print(f"\n=== 总计: {len(data)} 条 ===")
    src_counts = Counter(d.get("source", "unknown") for d in data)
    for src, cnt in src_counts.most_common():
        print(f"  {src}: {cnt}")

    out_path = PROJECT_ROOT / args.out
    with open(out_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已保存: {out_path}")

    # 抽检前 50 条
    check_path = out_path.parent / (out_path.stem + "_check.jsonl")
    with open(check_path, "w", encoding="utf-8") as f:
        for item in data[:50]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"抽检样本(前50条): {check_path}")


if __name__ == "__main__":
    main()