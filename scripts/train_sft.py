"""
阶段 1: QLoRA SFT 训练

在 Gemma-3-4B-IT 上用医疗指令数据做监督微调 (QLoRA),
训练完成后自动 merge LoRA adapter 并保存完整模型。

用法:
    python scripts/train_sft.py \
        --model google/gemma-3-4b-it \
        --data data/sft_medical.jsonl \
        --output_dir out/sft_medical \
        --epochs 2 --lr 2e-4 --r 64 --alpha 128 \
        --max-len 2048 --report-to swanlab
"""
import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def load_data(path):
    """加载 sft_medical.jsonl (instruction/input/output 格式)"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def format_chat(example, tokenizer):
    """把 instruction/input/output 转成 chat 格式文本"""
    instruction = example["instruction"]
    inp = example.get("input") or ""
    output = example["output"]
    user_msg = f"{instruction}\n{inp}" if inp else instruction
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": output},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", default="out/sft_medical")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--r", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
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

    # 加载模型和 tokenizer
    print(f"加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA 配置 (覆盖 q/k/v/o + gate/up/down)
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

    # 加载数据
    print(f"加载数据: {args.data}")
    rows = load_data(args.data)
    dataset = Dataset.from_list(rows)
    dataset = dataset.map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    print(f"训练样本: {len(dataset)} 条")

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        report_to=args.report_to,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_len,
        dataset_text_field="text",
        packing=False,
    )

    print("开始 SFT 训练 ...")
    trainer.train()

    # 保存 adapter
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter 已保存: {args.output_dir}")

    # merge adapter 到基础模型 (供 vLLM 直接加载评测)
    print("合并 LoRA adapter 到基础模型 ...")
    model = model.merge_and_unload()
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"完整模型已保存: {args.output_dir}")


if __name__ == "__main__":
    main()
