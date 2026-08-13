#!/bin/bash
# 阶段 1: SFT 训练一键脚本
# 用法: bash scripts/stage1_sft.sh
set -e

# 1. 准备数据
echo "===== [1/3] 准备 SFT 数据 ====="
python data/prepare_medical_data.py --total 60000 --out data/sft_medical.jsonl

# 2. SFT 训练
echo "===== [2/3] SFT 训练 (QLoRA) ====="
python scripts/train_sft.py \
    --model google/gemma-3-4b-it \
    --data data/sft_medical.jsonl \
    --output_dir out/sft_medical \
    --epochs 2 --lr 2e-4 --r 64 --alpha 128 \
    --max-len 2048 --report-to swanlab

# 3. 评测 SFT checkpoint
echo "===== [3/3] 评测 SFT checkpoint ====="
python evaluate/evaluate_medical.py \
    --model out/sft_medical \
    --benchmark cmb \
    --out results/sft_cmb.json

echo ""
echo "阶段 1 完成! 评测结果: results/sft_cmb.json"
