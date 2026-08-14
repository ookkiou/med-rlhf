"""
合并 LoRA adapter 到基础模型

QLoRA 训练时模型是 4bit 量化的, 无法直接 merge。
本脚本用 bf16 非量化方式加载基础模型, 挂载 adapter 后合并保存。

用法:
    python scripts/merge_lora.py
    python scripts/merge_lora.py --base google/gemma-3-4b-it --adapter out/sft_medical --out out/sft_medical_merged
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="google/gemma-3-4b-it",
                        help="基础模型 ID 或路径")
    parser.add_argument("--adapter", default="out/sft_medical",
                        help="LoRA adapter 路径")
    parser.add_argument("--out", default="out/sft_medical_merged",
                        help="合并后模型保存路径")
    args = parser.parse_args()

    print(f"加载基础模型 (bf16): {args.base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    print(f"加载 LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    print("合并 LoRA 权重到基础模型 ...")
    model = model.merge_and_unload()

    print(f"保存合并后的模型: {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)

    print("合并完成!")


if __name__ == "__main__":
    main()
