#!/bin/bash
# ============================================================
# 阶段 0: 环境准备 + 基线评测
#
# 用法: bash scripts/stage0_env_and_baseline.sh
#
# 执行内容:
#   0. AutoDL 初始化 (HF 镜像 + 登录 + PyTorch 安全升级 + CMExam 下载)
#   1. 环境检查 (Python/GPU/QLoRA/中文输出/前向反向)
#   2. 基座评测 - CMB (快速测试 100 条)
#   3. 基座评测 - CMB (全量)
#   4. 基座评测 - CMExam (如果数据存在)
#
# 注意:
#   - 首次运行会自动调用 setup_autodl.sh 初始化环境
#   - Gemma-3 是 gated model, 需要先在 HuggingFace 网站同意协议
#   - 如果 CMExam 数据未下载, 会跳过 CMExam 评测
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_NAME="${MODEL_NAME:-google/gemma-3-4b-it}"

echo "============================================================"
echo "  阶段 0: 环境准备 + 基线评测"
echo "  模型: $MODEL_NAME"
echo "============================================================"

# ------------------------------------------------------------
# Step 0: AutoDL 初始化 (首次运行)
# ------------------------------------------------------------
echo ""
echo "[Step 0] AutoDL 初始化..."
if [ ! -f ".setup_done" ]; then
    echo "  首次运行, 执行初始化..."
    bash scripts/setup_autodl.sh
    touch .setup_done
    echo "  ✓ 初始化完成"
else
    echo "  已初始化, 跳过"
fi

# ------------------------------------------------------------
# Step 1: 环境检查
# ------------------------------------------------------------
echo ""
echo "[Step 1] 环境检查..."
python scripts/check_env.py --model "$MODEL_NAME"

# ------------------------------------------------------------
# Step 2: 基座快速测试 - CMB (100 条, 验证流程能跑通)
# ------------------------------------------------------------
echo ""
echo "[Step 2] 基座快速测试 - CMB (100 条)..."
python evaluate/evaluate_medical.py \
    --model "$MODEL_NAME" \
    --benchmark cmb \
    --sample-size 100 \
    --out results/baseline_cmb_test.json

echo ""
echo "  快速测试完成, 查看结果:"
echo "    cat results/baseline_cmb_test.json | python -m json.tool | head -20"

# ------------------------------------------------------------
# Step 3: 基座全量评测 - CMB
# ------------------------------------------------------------
echo ""
echo "[Step 3] 基座全量评测 - CMB..."
read -p "  快速测试通过? 继续全量评测? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python evaluate/evaluate_medical.py \
        --model "$MODEL_NAME" \
        --benchmark cmb \
        --out results/baseline_cmb.json
else
    echo "  跳过全量评测"
fi

# ------------------------------------------------------------
# Step 4: 基座评测 - CMExam (如果数据存在)
# ------------------------------------------------------------
echo ""
echo "[Step 4] 基座评测 - CMExam..."
if [ -f "data/cmexam/val.csv" ] || [ -f "data/cmexam/test_with_annotations.csv" ] || [ -f "data/cmexam/test.csv" ] || [ -f "data/cmexam/test.json" ]; then
    # 快速测试 100 条
    python evaluate/evaluate_medical.py \
        --model "$MODEL_NAME" \
        --benchmark cmexam \
        --sample-size 100 \
        --out results/baseline_cmexam_test.json
    echo ""
    echo "  CMExam 快速测试完成"

    # 全量评测
    python evaluate/evaluate_medical.py \
        --model "$MODEL_NAME" \
        --benchmark cmexam \
        --out results/baseline_cmexam.json
    echo "  CMExam 全量评测完成: results/baseline_cmexam.json"
else
    echo "  ⚠ CMExam 数据未下载, 跳过"
    echo "    下载方法: bash scripts/setup_autodl.sh"
    echo "    或手动从 https://github.com/williamliujl/CMExam 下载 CSV 到 data/cmexam/"
fi

# ------------------------------------------------------------
# 汇总
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "  阶段 0 完成!"
echo ""
echo "  基座评测结果:"
echo "    CMB (快速):   results/baseline_cmb_test.json"
if [ -f "results/baseline_cmb.json" ]; then
    echo "    CMB (全量):   results/baseline_cmb.json"
fi
if [ -f "results/baseline_cmexam_test.json" ]; then
    echo "    CMExam (快速): results/baseline_cmexam_test.json"
fi
echo ""
echo "  预期分数 (基座 Gemma-3-4B-IT):"
echo "    CMB:     ~38 分"
echo "    CMExam:  ~45 分"
echo ""
echo "  下一步: 阶段 1 - SFT 训练"
echo "============================================================"
