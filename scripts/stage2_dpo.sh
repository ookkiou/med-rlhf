#!/bin/bash
# 阶段 2: DPO 安全对齐
# 在 SFT 模型基础上用偏好对数据做 DPO 训练
# 训练完后合并 LoRA → 评测

set -e

MODEL="/root/autodl-tmp/med-rlhf/out/sft_medical_v3_merged"
DATA="data/dpo_all.jsonl"
OUTPUT="out/dpo_medical"
MERGED="out/dpo_medical_merged"

echo "================================================"
echo "  阶段 2: DPO 安全对齐训练"
echo "  模型: $MODEL"
echo "  数据: $DATA"
echo "================================================"

# 检查 SFT 模型是否存在
if [ ! -d "$MODEL" ]; then
    echo "错误: SFT 模型不存在: $MODEL"
    echo "请先完成阶段 1 SFT 训练和合并"
    exit 1
fi

# 检查 DPO 数据是否存在
if [ ! -f "$DATA" ]; then
    echo "错误: DPO 偏好对数据不存在: $DATA"
    echo "请先运行: python data/build_safety_preference.py"
    exit 1
fi

# DPO 训练
echo ""
echo "[1/3] DPO 训练 ..."
python scripts/train_dpo.py \
    --model "$MODEL" \
    --data "$DATA" \
    --output_dir "$OUTPUT" \
    --epochs 2 \
    --lr 5e-6 \
    --beta 0.1 \
    --r 32 --alpha 64 \
    --max-len 1024 \
    --batch-size 4 --grad-accum 16 \
    --report-to swanlab

# 合并 LoRA
echo ""
echo "[2/3] 合并 DPO LoRA adapter ..."
python scripts/merge_lora.py \
    --base "$MODEL" \
    --adapter "$OUTPUT" \
    --out "$MERGED"

# 评测
echo ""
echo "[3/3] 安全评测 ..."
echo "  --- SFT 模型 (DPO 前) ---"
python evaluate/evaluate_safety.py \
    --model "$MODEL" \
    --chbench-data data/chbench \
    --high-risk-data data/safety_eval.jsonl \
    --out results/sft_safety.json

echo "  --- DPO 模型 (DPO 后) ---"
python evaluate/evaluate_safety.py \
    --model "$MERGED" \
    --chbench-data data/chbench \
    --high-risk-data data/safety_eval.jsonl \
    --out results/dpo_safety.json

echo ""
echo "================================================"
echo "  DPO 训练完成!"
echo "  对比结果:"
echo "    SFT: results/sft_safety.json"
echo "    DPO: results/dpo_safety.json"
echo "================================================"
