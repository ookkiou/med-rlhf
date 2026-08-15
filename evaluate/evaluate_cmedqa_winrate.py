#!/usr/bin/env python3
"""
cMedQA2 开放问答评测 (LLM-judge pairwise win-rate)

评测思路:
    cMedQA2 是真实患者提问的中文医疗问答数据集 (测试集 4000 题, 附医生回答)。
    对同一批问题分别用 基座模型 / SFT模型 生成回答,
    再用 DeepSeek API 作为裁判做两两对比 (position-swap 两轮, 消除位置偏差),
    输出 SFT 相对基座的 win-rate。

    这比 BLEU/ROUGE 更能反映 SFT 对回答质量的实际提升。

用法 (三个子命令依次执行):
    # 1. 下载数据 + 抽样 300 题
    python evaluate/evaluate_cmedqa_winrate.py prepare --n 300

    # 2. 生成回答 (基座和 SFT 各跑一次, transformers 后端)
    python evaluate/evaluate_cmedqa_winrate.py generate \
        --model google/gemma-3-4b-it --tag base
    python evaluate/evaluate_cmedqa_winrate.py generate \
        --model out/sft_medical_v3_merged --tag sft_v3

    # 3. DeepSeek 评判
    export DEEPSEEK_API_KEY=sk-xxx
    python evaluate/evaluate_cmedqa_winrate.py judge \
        --base-tag base --sft-tag sft_v3

数据来源: https://github.com/zhangsheng93/cMedQA2
    question.zip / answer.zip / test_candidates.zip
"""

import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cmedqa2"
RESULTS_DIR = PROJECT_ROOT / "results"

GITHUB_RAW = "https://raw.githubusercontent.com/zhangsheng93/cMedQA2/master"
GITHUB_PROXY = "https://ghproxy.net/https://raw.githubusercontent.com/zhangsheng93/cMedQA2/master"

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

GEN_PROMPT = (
    "你是一名专业、耐心的医生助手。请针对下面患者的问题，"
    "给出准确、有帮助的回答。\n\n患者问题：{question}"
)


# ============================================================
# prepare: 下载数据 + 构造评测集
# ============================================================

def download_repo_file(filename, dest_dir):
    """从 cMedQA2 仓库下载文件, 主链接失败自动走镜像"""
    dest = dest_dir / filename
    if dest.exists():
        print(f"  已存在, 跳过下载: {filename}")
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    for base_url in (GITHUB_RAW, GITHUB_PROXY):
        try:
            url = f"{base_url}/{filename}"
            print(f"  下载: {url}")
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as e:
            print(f"  失败: {e}")
    print(f"  ✗ 请手动下载 {filename} 放到 {dest_dir}/")
    return None


def read_csv_flexible(path):
    """读 CSV, 自动尝试 utf-8 / gbk 编码"""
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法解析文件编码: {path}")


def pick_col(headers, candidates):
    low = {h.lower().strip(): h for h in headers}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def cmd_prepare(n, seed):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    q_zip = download_repo_file("question.zip", DATA_DIR)
    a_zip = download_repo_file("answer.zip", DATA_DIR)
    c_zip = download_repo_file("test_candidates.zip", DATA_DIR)
    if not (q_zip and a_zip and c_zip):
        sys.exit(1)

    for z in (q_zip, a_zip, c_zip):
        if z.suffix == ".zip":
            print(f"  解压: {z.name}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(DATA_DIR)

    # 定位解压出的 csv / txt
    q_csv = next((p for p in DATA_DIR.rglob("*.csv") if "question" in p.name.lower()), None)
    a_csv = next((p for p in DATA_DIR.rglob("*.csv") if "answer" in p.name.lower()), None)
    c_txt = next((p for p in DATA_DIR.rglob("*.txt") if "candidate" in p.name.lower()), None)
    if not (q_csv and a_csv):
        print(f"  ✗ 未找到 questions/answers CSV, 请检查 {DATA_DIR} 目录")
        sys.exit(1)
    print(f"  questions: {q_csv.name}, answers: {a_csv.name}, candidates: {c_txt.name if c_txt else '(无)'}")

    # 问题表: qid -> content
    questions = read_csv_flexible(q_csv)
    qid_col = pick_col(questions[0].keys(), ["qid", "question_id", "q_id", "id"])
    content_col = pick_col(questions[0].keys(), ["content", "question", "title", "text"])
    if not (qid_col and content_col):
        print(f"  ✗ question.csv 字段不匹配, 实际表头: {list(questions[0].keys())}")
        sys.exit(1)
    print(f"  问题 {len(questions)} 条, 字段: qid={qid_col}, content={content_col}")

    # 回答表: qid -> 最长回答 (作为参考答案)
    answers = read_csv_flexible(a_csv)
    a_qid_col = pick_col(answers[0].keys(), ["qid", "question_id", "q_id"])
    a_content_col = pick_col(answers[0].keys(), ["content", "answer", "text"])
    if not (a_qid_col and a_content_col):
        print(f"  ✗ answer.csv 字段不匹配, 实际表头: {list(answers[0].keys())}")
        sys.exit(1)
    print(f"  回答 {len(answers)} 条, 字段: qid={a_qid_col}, content={a_content_col}")

    ref_by_qid = {}
    for row in answers:
        qid = (row.get(a_qid_col) or "").strip()
        content = (row.get(a_content_col) or "").strip()
        if qid and content:
            cur = ref_by_qid.get(qid, "")
            if len(content) > len(cur):
                ref_by_qid[qid] = content

    # 测试集 qid 集合 (跳过表头行)
    test_qids = set()
    if c_txt and c_txt.exists():
        with open(c_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("question_id"):
                    continue
                parts = line.replace(",", " ").split()
                if parts:
                    test_qids.add(parts[0])
        print(f"  测试集 qid: {len(test_qids)} 个")

    # 组装评测集: 测试集内、有参考答案、问题足够长
    pool = []
    for row in questions:
        qid = (row.get(qid_col) or "").strip()
        content = (row.get(content_col) or "").strip()
        if test_qids and qid not in test_qids:
            continue
        ref = ref_by_qid.get(qid, "")
        if len(content) >= 10 and len(ref) >= 20:
            pool.append({
                "qid": qid,
                "question": content,
                "reference": ref[:800],
            })

    print(f"  符合条件的候选题: {len(pool)} 条")
    if n < len(pool):
        random.seed(seed)
        pool = random.sample(pool, n)
    print(f"  抽样 {len(pool)} 条 (seed={seed})")

    out = DATA_DIR / "eval_set.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for item in pool:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"评测集已保存: {out}")


# ============================================================
# generate: 用待评测模型生成回答
# ============================================================

def cmd_generate(model_path, tag, max_new_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    eval_set = load_eval_set()
    out_path = RESULTS_DIR / f"cmedqa_gen_{tag}.jsonl"

    # 断点续跑: 已生成的直接跳过
    done = {}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    done[item["qid"]] = item
    todo = [q for q in eval_set if q["qid"] not in done or not done[q["qid"]]["model_answer_text"]]
    print(f"待生成: {len(todo)}/{len(eval_set)} (已完成 {len(done)})")

    if todo:
        print(f"加载模型 (transformers 后端): {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            for i, q in enumerate(todo):
                messages = [{"role": "user", "content": GEN_PROMPT.format(question=q["question"])}]
                text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                    )
                gen = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                done[q["qid"]] = {"qid": q["qid"], "model_answer_text": gen.strip()}
                f.write(json.dumps(done[q["qid"]], ensure_ascii=False) + "\n")
                f.flush()
                if (i + 1) % 20 == 0 or i + 1 == len(todo):
                    print(f"  进度: {i + 1}/{len(todo)}")

        del model
        torch.cuda.empty_cache()

    print(f"生成结果已保存: {out_path}")


def load_eval_set():
    path = DATA_DIR / "eval_set.jsonl"
    if not path.exists():
        print("✗ 评测集不存在, 请先运行 prepare 子命令")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_gen(tag):
    path = RESULTS_DIR / f"cmedqa_gen_{tag}.jsonl"
    if not path.exists():
        print(f"✗ 生成结果不存在: {path} (先运行 generate --tag {tag})")
        sys.exit(1)
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                out[item["qid"]] = item["model_answer_text"]
    return out


# ============================================================
# judge: DeepSeek 两两对比 (position-swap)
# ============================================================

JUDGE_PROMPT = """你是一名医疗问答质量评估专家。下面有患者的问题、医生提供的参考答案，以及两个 AI 模型生成的回答。

【患者问题】
{question}

【参考答案（来自真实医生）】
{reference}

【回答 A】
{answer_a}

【回答 B】
{answer_b}

请依据参考答案，从以下维度评估两个回答的质量：
1. 医学正确性：是否有事实错误或编造信息
2. 完整性：是否切中患者的问题
3. 帮助性与安全性：建议是否合理、是否可能造成伤害

注意：不要根据回答长度或格式华丽程度评判，只关注内容质量。若两者质量相当则判平局。

请严格按以下格式输出，第一行必须是判定行，不要输出其他内容：
判定: A 或 B 或 TIE
理由: 一句话"""


def call_deepseek(messages, api_key, model):
    import requests

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 300,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(4):
        try:
            r = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=90)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            wait = 3 * (attempt + 1)
            print(f"    API 调用失败 ({e}), {wait}s 后重试...")
            time.sleep(wait)
    return None


def parse_verdict(text):
    """解析 '判定: A/B/TIE'，解析失败返回 None"""
    if not text:
        return None
    m = re.search(r"判定\s*[:：]?\s*\**\s*(TIE|A|B|平局|持平)", text)
    if m:
        v = m.group(1)
        if v in ("TIE", "平局", "持平"):
            return "TIE"
        return v
    return None


def cmd_judge(base_tag, sft_tag, api_key, model, limit):
    eval_set = load_eval_set()
    base_gen = load_gen(base_tag)
    sft_gen = load_gen(sft_tag)

    pairs = [
        q for q in eval_set
        if q["qid"] in base_gen and q["qid"] in sft_gen
        and base_gen[q["qid"]].strip() and sft_gen[q["qid"]].strip()
    ]
    if limit:
        pairs = pairs[:limit]
    print(f"可评判题目: {len(pairs)} (judge={model})")

    details = []
    sft_win = sft_loss = tie = 0

    for i, q in enumerate(pairs):
        base_ans = base_gen[q["qid"]]
        sft_ans = sft_gen[q["qid"]]
        rounds = []

        # 两轮: 第一轮 SFT 为 A, 第二轮 SFT 为 B (position-swap 消除位置偏差)
        for sft_as in ("A", "B"):
            a_text = sft_ans if sft_as == "A" else base_ans
            b_text = base_ans if sft_as == "A" else sft_ans
            prompt = JUDGE_PROMPT.format(
                question=q["question"], reference=q["reference"],
                answer_a=a_text[:2000], answer_b=b_text[:2000],
            )
            reply = call_deepseek(
                [{"role": "user", "content": prompt}], api_key, model)
            verdict = parse_verdict(reply)
            if verdict is None:
                verdict = "TIE"  # 解析失败按平局处理
            # 换算到 SFT 视角
            if verdict == "TIE":
                rounds.append("TIE")
            elif verdict == sft_as:
                rounds.append("WIN")
            else:
                rounds.append("LOSS")

        # 两轮合成: 两轮都赢 → 赢; 一赢一平 → 赢; 一赢一输 / 两平 → 平; 否则输
        wins = rounds.count("WIN")
        losses = rounds.count("LOSS")
        if wins > losses:
            final = "WIN"
            sft_win += 1
        elif losses > wins:
            final = "LOSS"
            sft_loss += 1
        else:
            final = "TIE"
            tie += 1

        details.append({"qid": q["qid"], "rounds": rounds, "final": final})
        if (i + 1) % 20 == 0 or i + 1 == len(pairs):
            print(f"  进度: {i + 1}/{len(pairs)} | SFT胜 {sft_win} 平 {tie} 负 {sft_loss}")

    n = len(pairs)
    result = {
        "benchmark": "cmedqa2",
        "judge_model": model,
        "base": base_tag,
        "sft": sft_tag,
        "total": n,
        "sft_win": sft_win,
        "tie": tie,
        "sft_loss": sft_loss,
        "win_rate": f"{sft_win / n * 100:.1f}%" if n else "N/A",
        "decisive_win_rate": (
            f"{sft_win / (sft_win + sft_loss) * 100:.1f}%" if (sft_win + sft_loss) else "N/A"
        ),
        "details": details,
    }

    out_path = RESULTS_DIR / f"cmedqa_winrate_{base_tag}_vs_{sft_tag}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  cMedQA2 LLM-judge 结果: {sft_tag} vs {base_tag}")
    print(f"  SFT 胜: {sft_win} | 平: {tie} | 负: {sft_loss} (共 {n})")
    print(f"  win_rate (含平局): {result['win_rate']}")
    print(f"  decisive_win_rate (去平局): {result['decisive_win_rate']}")
    print(f"{'=' * 60}")
    print(f"结果已保存: {out_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="cMedQA2 win-rate 评测")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare", help="下载数据并构造评测集")
    p_prepare.add_argument("--n", type=int, default=300)
    p_prepare.add_argument("--seed", type=int, default=42)

    p_gen = sub.add_parser("generate", help="用模型生成回答")
    p_gen.add_argument("--model", required=True)
    p_gen.add_argument("--tag", required=True, help="输出标记, 如 base / sft_v3")
    p_gen.add_argument("--max-new-tokens", type=int, default=512)

    p_judge = sub.add_parser("judge", help="DeepSeek 两两对比评判")
    p_judge.add_argument("--base-tag", required=True)
    p_judge.add_argument("--sft-tag", required=True)
    p_judge.add_argument("--judge-model", default=DEEPSEEK_MODEL)
    p_judge.add_argument("--limit", type=int, default=None, help="只评判前 N 题 (调试用)")
    p_judge.add_argument("--api-key", default=None, help="默认读环境变量 DEEPSEEK_API_KEY")

    args = parser.parse_args()

    if args.cmd == "prepare":
        cmd_prepare(args.n, args.seed)
    elif args.cmd == "generate":
        cmd_generate(args.model, args.tag, args.max_new_tokens)
    elif args.cmd == "judge":
        import os
        key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            print("✗ 未设置 DEEPSEEK_API_KEY 环境变量")
            sys.exit(1)
        cmd_judge(args.base_tag, args.sft_tag, key, args.judge_model, args.limit)


if __name__ == "__main__":
    main()
