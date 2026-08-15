#!/usr/bin/env python3
"""
CMB / CMExam 医疗评测脚本

对基座 / SFT / DPO / PPO checkpoint 在 CMB 和 CMExam 两个公开医疗 benchmark 上评测。
使用 vLLM 批量推理, 计算选择题准确率。

用法:
    # 基座评测 (Gemma-3-IT 默认用 chat template)
    python evaluate/evaluate_medical.py \\
        --model google/gemma-3-4b-it \\
        --benchmark cmb \\
        --out results/baseline_cmb.json

    # SFT checkpoint 评测
    python evaluate/evaluate_medical.py \\
        --model out/sft_medical \\
        --benchmark cmb \\
        --chat-template \\
        --out results/sft_cmb.json

    # 快速测试 (采样 100 条)
    python evaluate/evaluate_medical.py \\
        --model google/gemma-3-4b-it \\
        --benchmark cmb \\
        --sample-size 100 \\
        --out results/baseline_cmb_test.json

注意:
    - Gemma-3-4B-IT 是多模态模型, vLLM 加载时需要 limit_mm_per_prompt={"image": 0}
    - CMB 数据集的 split 是 "val" (有答案), 不是 "test" (无答案)
    - IT 模型默认使用 chat template, 加 --no-chat-template 可关闭
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# 数据加载
# ============================================================

def load_cmb(split="val"):
    """
    加载 CMB-Exam 数据集 (HuggingFace: FreedomIntelligence/CMB)

    CMB-Exam splits:
        - train: 269,359 题 (无答案, 用于知识注入)
        - val: 280 题 (有答案和解析, 用于评测)
        - test: 11,200 题 (有答案, 但主要用于提交排行榜)

    我们用 val 做评测 (有答案 + 有解析, 适合 few-shot/CoT)

    数据格式:
        {
            "exam_type": "医师考试",
            "exam_class": "执业医师",
            "exam_subject": "口腔执业医师",
            "question": "患者,男性,11岁...",
            "answer": "D",
            "question_type": "单项选择题",
            "option": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."}
        }

    注意: 直接用 load_dataset 在 datasets 5.x 下会报 schema 校验错误,
    这里改用 hf_hub_download 下载原始 JSON 手动解析, 更稳定。
    """
    from huggingface_hub import hf_hub_download

    print(f"加载 CMB-Exam 数据集 (split={split})...")

    # CMB-Exam 原始文件名映射 (注意: 文件在 CMB-Exam/<split>/ 子目录下)
    file_map = {
        "val": "CMB-Exam/CMB-val/CMB-val-merge.json",
        "test": "CMB-Exam/CMB-test/CMB-test-choice-question-merge.json",
        "train": "CMB-Exam/CMB-train/CMB-train-merge.json",
    }
    if split not in file_map:
        print(f"  split '{split}' 不存在, 使用 'val' (有答案的评测集)")
        split = "val"
    file_name = file_map[split]

    # 下载原始 JSON (自动缓存到 HF_HOME)
    local_path = hf_hub_download(
        repo_id="FreedomIntelligence/CMB",
        filename=file_name,
        repo_type="dataset",
    )
    print(f"  数据文件: {local_path}")

    with open(local_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 标准化: option 字段可能是字符串形式的 JSON, 解析成 dict
    data = []
    for item in raw:
        item = dict(item)
        opt = item.get("option")
        if isinstance(opt, str):
            try:
                item["option"] = json.loads(opt)
            except Exception:
                item["option"] = {}
        data.append(item)

    print(f"  加载 {len(data)} 条数据")
    return data


def load_cmexam(data_path=None):
    """
    加载 CMExam 数据集

    CMExam 来自中国国家医师考试, 包含 60K+ 选择题。
    数据下载: https://github.com/williamliujl/CMExam

    支持格式:
        - CSV (原始格式: train.csv / val.csv / test_with_annotations.csv)
        - JSON 数组, 每条包含 question, answer, options
        - JSONL, 每行一条

    CSV 字段 (CMExam 原始):
        Question, Options (JSON字符串), Answer, Explanation, ...
    """
    if data_path is None:
        # 尝试多个可能的路径 (优先 CSV, 兼容 JSON)
        candidates = [
            PROJECT_ROOT / "data" / "cmexam" / "val.csv",
            PROJECT_ROOT / "data" / "cmexam" / "test_with_annotations.csv",
            PROJECT_ROOT / "data" / "cmexam" / "test.csv",
            PROJECT_ROOT / "data" / "cmexam" / "test.json",
            PROJECT_ROOT / "data" / "cmexam" / "val.json",
            PROJECT_ROOT / "data" / "cmexam" / "test.jsonl",
        ]
        for p in candidates:
            if p.exists():
                data_path = p
                break
        else:
            data_path = PROJECT_ROOT / "data" / "cmexam" / "val.csv"

    data_path = Path(data_path)
    if not data_path.exists():
        print(f"  ✗ CMExam 数据文件不存在: {data_path}")
        print(f"  请从 https://github.com/williamliujl/CMExam 下载数据")
        print(f"  或运行: bash scripts/setup_autodl.sh")
        print(f"  下载后放到 data/cmexam/ 目录下")
        return None

    print(f"加载 CMExam 数据: {data_path}")

    data = []
    if data_path.suffix == ".csv":
        import csv

        with open(data_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # CMExam CSV 字段名首字母大写或小写不一, 兼容处理
                question = row.get("Question") or row.get("question") or ""
                answer = row.get("Answer") or row.get("answer") or ""
                options_raw = row.get("Options") or row.get("options") or ""

                # Options 在 CSV 里是 JSON 字符串, 如 {"A": "...", "B": "..."}
                # 也可能是用 | 分隔的纯文本
                if options_raw:
                    try:
                        options = json.loads(options_raw)
                    except (json.JSONDecodeError, TypeError):
                        # 尝试按 | 或换行分割
                        parts = options_raw.split("|") if "|" in options_raw else options_raw.split("\n")
                        options = {}
                        letters = "ABCDE"
                        for i, part in enumerate(parts):
                            if i < len(letters):
                                options[letters[i]] = part.strip()
                else:
                    options = {}

                data.append({
                    "question": question,
                    "answer": answer.strip().upper() if answer else "",
                    "options": options,
                })
    elif data_path.suffix == ".jsonl":
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    print(f"  加载 {len(data)} 条数据")
    return data


MMLU_MED_SUBJECTS = [
    "anatomy",              # 解剖学
    "clinical_knowledge",   # 临床知识
    "college_biology",      # 大学生物
    "college_medicine",     # 大学医学
    "medical_genetics",     # 医学遗传学
    "professional_medicine",# 职业医学
    "virology",             # 病毒学
    "nutrition",            # 营养学
]


def load_mmlu_med(subjects=None, split="test"):
    """
    加载 MMLU 医学相关子集 (HuggingFace: cais/mmlu)

    共 8 个子集 test split 合计约 1500+ 题, 英文 4 选 1。
    用途: 检验中文医疗 SFT 后通用医学能力是否回退 (预期持平, 非提升指标)。
    """
    from datasets import load_dataset

    if subjects is None:
        subjects = MMLU_MED_SUBJECTS

    data = []
    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj, split=split)
        letters = "ABCD"
        for row in ds:
            data.append({
                "question": row["question"],
                "options": {letters[i]: c for i, c in enumerate(row["choices"])},
                "answer": letters[row["answer"]],
                "subject": subj,
            })
        print(f"  {subj}: {len(ds)} 题")

    print(f"  加载 {len(data)} 条数据 (MMLU 医学子集 x {len(subjects)})")
    return data


# ============================================================
# Prompt 构造
# ============================================================

# CMB few-shot 示例 (帮助模型理解输出格式: 单选一个字母, 多选多个字母)
CMB_FEWSHOT_EXAMPLES = """以下是中国医师考试中的一道单项选择题,不需要做任何分析和解释,直接输出答案选项。
患者,男,60岁,突发胸骨后压榨性疼痛3小时,含服硝酸甘油不缓解,心电图示V1-V4导联ST段弓背向上抬高。最可能的诊断是?
A. 急性心肌梗死
B. 稳定型心绞痛
C. 主动脉夹层
D. 急性肺栓塞
答案: A

以下是中国医师考试中的一道单项选择题,不需要做任何分析和解释,直接输出答案选项。
患者,女,30岁,尿频、尿急、尿痛2天,伴发热,尿常规示白细胞满视野,亚硝酸盐阳性。最可能的诊断是?
A. 急性肾小球肾炎
B. 急性膀胱炎
C. 肾结石
D. 肾病综合征
答案: B

以下是中国医师考试中的一道多项选择题,不需要做任何分析和解释,直接输出答案选项。
关于高血压降压治疗,下列正确的有?
A. 合并糖尿病者首选ACEI/ARB
B. 老年单纯收缩期高血压首选CCB
C. 合并心力衰竭者禁用β受体阻滞剂
D. 妊娠期高血压首选ACEI
E. 高血压急症首选口服短效降压药
答案: AB

"""


def build_cmb_prompt(item):
    """
    构造 CMB-Exam 评测 prompt

    格式参考 CMB 官方:
        以下是中国{exam_type}中{exam_class}考试的一道{question_type},
        不需要做任何分析和解释, 直接输出答案选项。
        {题目}
        A. {选项A}
        B. {选项B}
        ...

    前面拼接 few-shot 示例, 引导模型输出格式 (单选一个字母, 多选多个字母)。
    """
    item = dict(item) if hasattr(item, "items") else item
    exam_type = item.get("exam_type", "医师考试")
    exam_class = item.get("exam_class", "执业医师")
    question_type = item.get("question_type", "单项选择题")
    question = item["question"]
    options = item.get("option", {})

    option_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))

    prompt = (
        CMB_FEWSHOT_EXAMPLES
        + f"以下是中国{exam_type}中{exam_class}考试的一道{question_type},"
        f"不需要做任何分析和解释,直接输出答案选项。\n"
        f"{question}\n"
        f"{option_text}\n"
        f"答案: "
    )

    return prompt


# CMExam few-shot 示例
CMEXAM_FEWSHOT_EXAMPLES = """以下是中国国家医师考试中的一道单项选择题,不需要做任何分析和解释,直接输出答案选项。
患者,男,60岁,突发胸骨后压榨性疼痛3小时,含服硝酸甘油不缓解,心电图示V1-V4导联ST段弓背向上抬高。最可能的诊断是?
A. 急性心肌梗死
B. 稳定型心绞痛
C. 主动脉夹层
D. 急性肺栓塞
答案: A

以下是中国国家医师考试中的一道单项选择题,不需要做任何分析和解释,直接输出答案选项。
患者,女,30岁,尿频、尿急、尿痛2天,伴发热,尿常规示白细胞满视野,亚硝酸盐阳性。最可能的诊断是?
A. 急性肾小球肾炎
B. 急性膀胱炎
C. 肾结石
D. 肾病综合征
答案: B

"""


def build_cmexam_prompt(item):
    """构造 CMExam 评测 prompt"""
    item = dict(item) if hasattr(item, "items") else item
    question = item.get("question", "")

    if "options" in item and item["options"]:
        options = item["options"]
        option_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        prompt = (
            CMEXAM_FEWSHOT_EXAMPLES
            + f"以下是中国国家医师考试中的一道选择题,"
            f"不需要做任何分析和解释,直接输出答案选项。\n"
            f"{question}\n"
            f"{option_text}\n"
            f"答案: "
        )
    else:
        prompt = (
            CMEXAM_FEWSHOT_EXAMPLES
            + f"以下是中国国家医师考试中的一道选择题,"
            f"不需要做任何分析和解释,直接输出答案选项。\n"
            f"{question}\n"
            f"答案: "
        )

    return prompt


def build_mmlu_prompt(item):
    """构造 MMLU 英文医学选择题 prompt (0-shot, 要求只输出字母)"""
    item = dict(item) if hasattr(item, "items") else item
    question = item.get("question", "")
    options = item.get("options", {})
    option_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return (
        "The following is a multiple choice question about medical knowledge. "
        "Answer with the option letter only, no explanation.\n\n"
        f"{question}\n"
        f"{option_text}\n"
        f"Answer: "
    )


# ============================================================
# 答案提取
# ============================================================

def extract_answer(text, options=None):
    """
    从模型输出中提取答案选项 (支持多选)

    支持多种格式:
        - "A" / "B" / "C" / "D" / "E"
        - "BCDE" / "A, C, E" / "A C E"
        - "答案是A" / "答案为A" / "答案:A"
        - "选A" / "选择A"
        - "A选项" / "选项A"
        - 开头的 "A." / "A、"

    返回排序去重后的字母字符串, 如 "BCDE" (多选) 或 "B" (单选)。
    """
    valid_options = set(options.keys()) if options else {"A", "B", "C", "D", "E"}

    # 1. 关键词后的字母序列 (最可靠): "答案: BCDE", "选 A, C, E"
    for pattern in [
        r"答案[是为：:]\s*([A-E][A-E,\s、，]*)",
        r"选[择]?\s*([A-E][A-E,\s、，]*)",
        r"选项\s*([A-E][A-E,\s、，]*)",
    ]:
        match = re.search(pattern, text)
        if match:
            letters = [c for c in re.findall(r"[A-E]", match.group(1)) if c in valid_options]
            if letters:
                return "".join(sorted(set(letters)))

    # 2. 独立字母 token: "B. 内容", "A, C, E"
    letters = [c for c in re.findall(r"\b[A-E]\b", text) if c in valid_options]
    if letters:
        return "".join(sorted(set(letters)))

    # 3. 连续字母串: "BCDE"
    letters = [c for c in re.findall(r"[A-E]", text) if c in valid_options]
    if letters:
        return "".join(sorted(set(letters)))

    return ""


# ============================================================
# vLLM 推理
# ============================================================

def run_vllm_inference(model_path, prompts, chat_template=False, max_new_tokens=64):
    """使用 vLLM 进行批量推理

    注意: Gemma-3-4B-IT 是多模态模型, vLLM 加载时需要:
    1. limit_mm_per_prompt={"image": 0} - 禁用图像输入
    2. 较新版本的 vLLM (>= 0.6.0) 支持 Gemma-3
    """
    from vllm import LLM, SamplingParams

    print(f"加载模型: {model_path}")

    # Gemma-3 多模态模型需要禁用图像输入
    llm_kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 2048,
    }

    # 尝试添加 limit_mm_per_prompt (Gemma-3 多模态需要)
    try:
        llm = LLM(**llm_kwargs, limit_mm_per_prompt={"image": 0})
    except TypeError:
        # 如果 vLLM 版本不支持 limit_mm_per_prompt, 尝试不带该参数
        print("  (vLLM 版本不支持 limit_mm_per_prompt, 尝试普通加载)")
        llm = LLM(**llm_kwargs)
    except Exception as e:
        # 如果多模态加载失败, 尝试使用 transformers 后端
        print(f"  (vLLM 原生加载失败: {e})")
        print(f"  (尝试使用 transformers 后端...)")
        llm = LLM(**llm_kwargs, model_impl="transformers")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop=["\n\n", "问：", "问:"],
    )

    if chat_template:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processed_prompts = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            processed_prompts.append(text)
        prompts = processed_prompts

    print(f"批量推理 {len(prompts)} 条...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        generated_text = output.outputs[0].text
        results.append(generated_text)

    return results


def run_transformers_inference(model_path, prompts, chat_template=False, max_new_tokens=64):
    """使用纯 transformers 进行批量推理 (不依赖 vLLM)

    当 vLLM 与 torch 版本不兼容时使用此后端。
    速度比 vLLM 慢, 但兼容性最好。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"加载模型 (transformers 后端): {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # 确定 EOS token
    eos_token_id = tokenizer.eos_token_id
    # 生成时的停止 token
    stop_strings = ["\n\n", "问：", "问:"]
    stop_token_ids = []
    for s in stop_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            stop_token_ids.append(ids[-1])
    if eos_token_id is not None:
        stop_token_ids.append(eos_token_id)

    results = []
    total = len(prompts)

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            if chat_template:
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            else:
                text = prompt

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 逐 token 生成, 遇到停止 token 就停
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=eos_token_id,
                eos_token_id=list(set(stop_token_ids)) if stop_token_ids else None,
            )

            # 只取新生成的 token
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0][input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # 去掉停止字符串之后的内容
            for stop_str in stop_strings:
                if stop_str in generated_text:
                    generated_text = generated_text[:generated_text.index(stop_str)]

            results.append(generated_text.strip())

            if (i + 1) % 20 == 0 or i + 1 == total:
                print(f"  进度: {i + 1}/{total}")

    # 释放显存
    del model
    torch.cuda.empty_cache()

    return results


# ============================================================
# 评测主流程
# ============================================================

def evaluate(
    model_path,
    benchmark,
    data_path=None,
    chat_template=False,
    max_new_tokens=64,
    out_path=None,
    sample_size=None,
    split="val",
    backend="auto",
):
    """评测主流程"""
    # 1. 加载数据
    if benchmark == "cmb":
        data = load_cmb(split=split)
        build_prompt = build_cmb_prompt
    elif benchmark == "cmexam":
        data = load_cmexam(data_path)
        if data is None:
            return None
        build_prompt = build_cmexam_prompt
    elif benchmark == "mmlu_med":
        data = load_mmlu_med()
        build_prompt = build_mmlu_prompt
    else:
        print(f"未知 benchmark: {benchmark}")
        return None

    # 采样
    if sample_size and sample_size < len(data):
        print(f"采样 {sample_size} 条用于快速测试...")
        if benchmark == "mmlu_med":
            # MMLU 是多个子集聚合, 用随机采样保证子集覆盖 (固定 seed 可复现)
            import random as _random
            _random.seed(42)
            data = _random.sample(list(data), sample_size)
        elif hasattr(data, "select"):
            data = data.select(range(sample_size))
        else:
            data = data[:sample_size]

    # 2. 构造 prompts
    print(f"构造 prompts...")
    prompts = []
    ground_truth = []
    for item in data:
        if hasattr(item, "items"):
            item = dict(item)
        prompt = build_prompt(item)
        prompts.append(prompt)
        ground_truth.append(item.get("answer", ""))

    # 3. 推理
    print(f"\n开始推理...")
    responses = None

    if backend in ("auto", "vllm"):
        try:
            responses = run_vllm_inference(
                model_path, prompts, chat_template=chat_template, max_new_tokens=max_new_tokens
            )
        except Exception as e:
            print(f"\n  vLLM 推理失败: {e}")
            if backend == "vllm":
                raise
            print(f"  自动回退到 transformers 后端 (较慢但兼容)...\n")
            responses = None

    if responses is None:
        responses = run_transformers_inference(
            model_path, prompts, chat_template=chat_template, max_new_tokens=max_new_tokens
        )

    # 4. 提取答案并计算准确率
    print(f"\n提取答案...")
    predictions = []
    correct = 0
    for i, (response, gt) in enumerate(zip(responses, ground_truth)):
        item = data[i]
        if hasattr(item, "items"):
            item = dict(item)
        options = item.get("option", item.get("options", {}))

        pred = extract_answer(response, options)
        # 多选比对: 标准答案 "BCDE" 和提取答案按集合比较 (顺序无关)
        is_correct = bool(pred) and set(pred) == set(gt)
        if is_correct:
            correct += 1

        predictions.append(
            {
                "id": i,
                "model_answer": pred,
                "ground_truth": gt,
                "model_output": response[:200],
                "correct": is_correct,
            }
        )

    accuracy = correct / len(ground_truth) if ground_truth else 0

    # 5. 输出结果
    result = {
        "model": model_path,
        "benchmark": benchmark,
        "total": len(ground_truth),
        "correct": correct,
        "accuracy": f"{accuracy * 100:.2f}%",
        "accuracy_raw": accuracy,
        "chat_template": chat_template,
        "predictions": predictions,
    }

    print(f"\n{'=' * 60}")
    print(f"  评测结果: {model_path} on {benchmark}")
    print(f"  准确率: {correct}/{len(ground_truth)} = {accuracy * 100:.2f}%")
    print(f"  chat_template: {chat_template}")
    print(f"{'=' * 60}")

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="CMB/CMExam 医疗评测")
    parser.add_argument(
        "--model", type=str, required=True, help="模型路径或 HuggingFace ID"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["cmb", "cmexam", "mmlu_med"],
        required=True,
        help="评测基准",
    )
    parser.add_argument(
        "--data-path", type=str, default=None, help="CMExam 数据路径"
    )
    parser.add_argument(
        "--chat-template", action="store_true", default=True,
        help="使用 chat template (默认开启, Gemma-3-IT 需要)"
    )
    parser.add_argument(
        "--no-chat-template", action="store_false", dest="chat_template",
        help="关闭 chat template"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64, help="最大生成 token 数"
    )
    parser.add_argument(
        "--out", type=str, default=None, help="输出文件路径"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None, help="采样数量 (用于快速测试)"
    )
    parser.add_argument(
        "--split", type=str, default="val",
        help="CMB 数据集 split (val 有答案, test 用于排行榜提交)"
    )
    parser.add_argument(
        "--backend", type=str, default="auto",
        choices=["auto", "vllm", "transformers"],
        help="推理后端: auto (默认, 先试vLLM失败回退transformers), vllm, transformers"
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        benchmark=args.benchmark,
        data_path=args.data_path,
        chat_template=args.chat_template,
        max_new_tokens=args.max_new_tokens,
        out_path=args.out,
        sample_size=args.sample_size,
        split=args.split,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
