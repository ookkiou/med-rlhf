#!/bin/bash
# 阶段 1 v3: SFT 训练（cMedQA2 医生回答骨干 + 保守参数防坍缩）
# 用法: nohup bash scripts/stage1_sft_v3.sh > logs/sft_v3.log 2>&1 &
# 查看日志: tail -f logs/sft_v3.log
set -e

mkdir -p logs

# 1. 准备数据（cMedQA2 医生回答 60% + CoT 25% + 通用 15%）
echo "===== [1/3] 准备 SFT 数据 v3 ====="
python data/prepare_medical_data_v3.py --total 8000 --out data/sft_medical_v3.jsonl

# 2. SFT 训练（沿用 v2 保守参数，防止输出坍缩）
echo "===== [2/3] SFT 训练 v3 (QLoRA) ====="
python scripts/train_sft.py \
    --model google/gemma-3-4b-it \
    --data data/sft_medical_v3.jsonl \
    --output_dir out/sft_medical_v3 \
    --epochs 1 --lr 1e-4 --r 32 --alpha 64 \
    --batch-size 4 --grad-accum 16 \
    --max-len 2048 --report-to swanlab

# 3. 评测 SFT checkpoint (CMB)
echo "===== [3/3] 评测 SFT v3 checkpoint ====="
python evaluate/evaluate_medical.py \
    --model out/sft_medical_v3 \
    --benchmark cmb \
    --out results/sft_v3_cmb.json

echo ""
echo "阶段 1 v3 完成! 评测结果: results/sft_v3_cmb.json"