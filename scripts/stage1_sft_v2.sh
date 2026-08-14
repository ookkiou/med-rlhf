#!/bin/bash
# 阶段 1 v2: SFT 训练（四源混合数据 + 保守参数防坍缩）
# 用法: nohup bash scripts/stage1_sft_v2.sh > logs/sft_v2.log 2>&1 &
# 查看日志: tail -f logs/sft_v2.log
set -e

mkdir -p logs

# 1. 准备数据（四源混合: QA 55% + 选择题 18% + CoT 22% + 通用 5%）
echo "===== [1/3] 准备 SFT 数据 v2 ====="
python data/prepare_medical_data_v2.py --total 10000 --out data/sft_medical_v2.jsonl

# 2. SFT 训练（降低 r 和 lr 防止输出坍缩）
echo "===== [2/3] SFT 训练 v2 (QLoRA) ====="
python scripts/train_sft.py \
    --model google/gemma-3-4b-it \
    --data data/sft_medical_v2.jsonl \
    --output_dir out/sft_medical_v2 \
    --epochs 1 --lr 1e-4 --r 32 --alpha 64 \
    --batch-size 4 --grad-accum 16 \
    --max-len 2048 --report-to swanlab

# 3. 评测 SFT checkpoint
echo "===== [3/3] 评测 SFT v2 checkpoint ====="
python evaluate/evaluate_medical.py \
    --model out/sft_medical_v2 \
    --benchmark cmb \
    --out results/sft_v2_cmb.json

echo ""
echo "阶段 1 v2 完成! 评测结果: results/sft_v2_cmb.json"
