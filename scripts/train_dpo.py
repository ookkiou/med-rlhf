"""
阶段 2: DPO 安全对齐训练

在 SFT 模型基础上用 DPO 偏好对数据训练, 让模型学会安全拒绝 + 引导就医。
使用 QLoRA + TRL DPOTrainer, 训练完后需用 merge_lora.py 合并。

用法:
    python scripts/train_dpo.py \
        --model out/sft_medical_v2_merged \
        --data data/dpo_train.jsonl \
        --output_dir out/dpo_medical \
        --epochs 2 --lr 5e-6 --beta 0.1 \
        --batch-size 4 --grad-accum 16 \
        --report-to swanlab
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOTrainer, DPOConfig


def load_dpo_data(path):
    """加载 dpo_train.jsonl (prompt/chosen/rejected 格式)"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="out/sft_medical_v2_merged",
                        help="SFT 模型路径 (DPO 起点)")
    parser.add_argument("--data", required=True,
                        help="DPO 偏好对数据路径")
    parser.add_argument("--output_dir", default="out/dpo_medical",
                        help="输出目录")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="DPO 学习率 (通常 1e-6 ~ 1e-5, 远低于 SFT)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO 温度参数 (0.1-0.5, 越大越保守)")
    parser.add_argument("--r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=64,
                        help="LoRA alpha")
    parser.add_argument("--max-len", type=int, default=1024,
                        help="最大序列长度 (DPO 数据较短, 1024 够用)")
    parser.add_argument("--max-prompt-len", type=int, default=256,
                        help="最大 prompt 长度")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--report-to", default="swanlab")
    args = parser.parse_args()

    # 4bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 加载 SFT 模型和 tokenizer
    print(f"加载 SFT 模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA 配置 (与 SFT 一致)
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 加载偏好对数据
    print(f"加载偏好对数据: {args.data}")
    rows = load_dpo_data(args.data)
    dataset = Dataset.from_list(rows)
    print(f"训练样本: {len(dataset)} 对偏好")

    # DPO 训练参数
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        report_to=args.report_to,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        beta=args.beta,
        max_length=args.max_len,
        max_prompt_length=args.max_prompt_len,
    )

    # SwanLab 初始化
    if args.report_to == "swanlab":
        import swanlab
        os.environ.pop("SWANLAB_PROJECT", None)
        os.environ.pop("SWANLAB_WORKSPACE", None)
        swanlab.init(
            project="med-rlhf",
            workspace="ookkiou",
        )

    # DPO Trainer
    # peft_config + 4bit 模型: DPOTrainer 会自动用冻结的 base model 作为 reference
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("开始 DPO 训练 ...")
    trainer.train()

    # 保存 adapter
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"DPO LoRA adapter 已保存: {args.output_dir}")
    print("如需合并为完整模型, 请运行:")
    print(f"  python scripts/merge_lora.py --base {args.model} --adapter {args.output_dir} --out out/dpo_medical_merged")


if __name__ == "__main__":
    main()
