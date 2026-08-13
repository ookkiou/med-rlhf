#!/bin/bash
# ============================================================
# AutoDL 服务器初始化脚本
#
# 解决以下问题:
#   1. HuggingFace 镜像设置 (国内加速)
#   2. HuggingFace 登录 (Gemma-3 是 gated model)
#   3. PyTorch 安全升级 (不破坏 CUDA)
#   4. CMExam 数据下载
#   5. 数据盘环境变量配置
#
# 用法: bash scripts/setup_autodl.sh
# ============================================================

set -e

echo "============================================================"
echo "  AutoDL 服务器初始化"
echo "============================================================"

# ------------------------------------------------------------
# Step 0: 数据盘配置
# ------------------------------------------------------------
echo ""
echo "[Step 0] 配置数据盘..."

# 设置 HuggingFace 缓存到数据盘
if [ -d "/root/autodl-tmp" ]; then
    DATA_DISK="/root/autodl-tmp"
else
    DATA_DISK="$HOME"
fi

echo "  数据盘: $DATA_DISK"

# 写入环境变量到 .bashrc (幂等)
write_env() {
    local key="$1"
    local val="$2"
    if ! grep -q "^export $key=" ~/.bashrc; then
        echo "export $key=$val" >> ~/.bashrc
    fi
}

write_env "HF_HOME" "$DATA_DISK/hf_cache"
write_env "HF_ENDPOINT" "https://hf-mirror.com"
write_env "HF_DATASETS_CACHE" "$DATA_DISK/hf_cache/datasets"

export HF_HOME="$DATA_DISK/hf_cache"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_DATASETS_CACHE="$DATA_DISK/hf_cache/datasets"

mkdir -p "$HF_HOME"
echo "  ✓ 环境变量已配置"
echo "    HF_HOME=$HF_HOME"
echo "    HF_ENDPOINT=$HF_ENDPOINT"

# ------------------------------------------------------------
# Step 1: PyTorch 版本检查 (安全升级, 不破坏 CUDA)
# ------------------------------------------------------------
echo ""
echo "[Step 1] 检查 PyTorch 版本..."

PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

echo "  Python: $PYTHON_VERSION"
echo "  PyTorch: $TORCH_VERSION"
echo "  CUDA available: $CUDA_AVAILABLE"

# 检查是否需要升级 PyTorch
NEED_UPGRADE=false
if python -c "import torch; assert torch.__version__ >= '2.2.0'" 2>/dev/null; then
    echo "  ✓ PyTorch 版本满足要求 (>= 2.2.0)"
else
    echo "  ⚠ PyTorch 版本 < 2.2.0, 需要升级"
    NEED_UPGRADE=true
fi

if [ "$NEED_UPGRADE" = true ]; then
    echo ""
    echo "  升级 PyTorch (CUDA 12.1 版本)..."
    echo "  注意: 使用 PyTorch 官方 index-url, 确保 CUDA 支持"
    pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu121

    # 验证升级后 CUDA 仍可用
    CUDA_CHECK=$(python -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_CHECK" = "True" ]; then
        echo "  ✓ PyTorch 升级成功, CUDA 可用"
    else
        echo "  ✗ PyTorch 升级后 CUDA 不可用!"
        echo "  请尝试更换 AutoDL 镜像为 PyTorch 2.3+ 版本"
        exit 1
    fi
fi

# ------------------------------------------------------------
# Step 2: 安装其他依赖 (排除 torch, 已单独处理)
# ------------------------------------------------------------
echo ""
echo "[Step 2] 安装项目依赖..."

# 先升级 transformers (Gemma-3 需要 >= 4.50)
pip install "transformers>=4.50.0"
pip install "peft>=0.11" "trl>=0.12" "bitsandbytes>=0.43" "accelerate>=0.30"
pip install "datasets>=2.19" "huggingface_hub>=0.23"
pip install "vllm>=0.6.0"
pip install sentencepiece protobuf
pip install wandb
pip install numpy pandas tqdm scikit-learn

echo "  ✓ 依赖安装完成"

# ------------------------------------------------------------
# Step 3: HuggingFace 登录
# ------------------------------------------------------------
echo ""
echo "[Step 3] HuggingFace 登录..."
echo ""
echo "  ⚠ Gemma-3 是 gated model, 需要授权:"
echo "    1. 访问 https://huggingface.co/google/gemma-3-4b-it"
echo "    2. 登录 HuggingFace 账号, 点击 'Agree and access repository'"
echo "    3. 获取 Access Token: https://huggingface.co/settings/tokens"
echo ""

# 检查是否已登录
if python -c "from huggingface_hub import HfFolder; HfFolder.load_token()" 2>/dev/null; then
    echo "  ✓ 已检测到 HuggingFace token"
else
    echo "  请输入 HuggingFace Access Token (或按 Ctrl+C 跳过, 稍后手动登录):"
    huggingface-cli login
fi

# ------------------------------------------------------------
# Step 4: 下载 CMExam 数据
# ------------------------------------------------------------
echo ""
echo "[Step 4] 下载 CMExam 数据..."

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CMEXAM_DIR="$PROJECT_ROOT/data/cmexam"
mkdir -p "$CMEXAM_DIR"

if [ -f "$CMEXAM_DIR/test.json" ]; then
    echo "  ✓ CMExam 数据已存在"
else
    echo "  从 GitHub 下载 CMExam 数据..."
    # CMExam 数据在 GitHub 仓库的 data 目录
    # 下载 test 集的 prompt 文件
    cd /tmp
    git clone --depth 1 https://github.com/williamliujl/CMExam.git 2>/dev/null || true

    if [ -d "/tmp/CMExam/data" ]; then
        # 查找 test 数据文件
        TEST_FILE=$(find /tmp/CMExam/data -name "*test*" -name "*.json" -o -name "*test*" -name "*.jsonl" | head -1)
        if [ -n "$TEST_FILE" ]; then
            cp "$TEST_FILE" "$CMEXAM_DIR/test.json"
            echo "  ✓ CMExam 数据已复制到 $CMEXAM_DIR/test.json"
        else
            echo "  ⚠ 未找到 CMExam test 文件, 请手动下载"
            echo "    仓库: https://github.com/williamliujl/CMExam"
            echo "    放到: $CMEXAM_DIR/test.json"
        fi
        rm -rf /tmp/CMExam
    else
        echo "  ⚠ CMExam 下载失败, 请手动下载"
        echo "    仓库: https://github.com/williamliujl/CMExam"
        echo "    放到: $CMEXAM_DIR/test.json"
    fi
fi

# ------------------------------------------------------------
# Step 5: 验证
# ------------------------------------------------------------
echo ""
echo "[Step 5] 验证关键依赖版本..."
python -c "
import torch, transformers, peft, trl, bitsandbytes, datasets
print(f'  torch:          {torch.__version__}')
print(f'  transformers:   {transformers.__version__}')
print(f'  peft:           {peft.__version__}')
print(f'  trl:            {trl.__version__}')
print(f'  bitsandbytes:   {bitsandbytes.__version__}')
print(f'  datasets:       {datasets.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:            {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory:     {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

echo ""
echo "============================================================"
echo "  初始化完成!"
echo ""
echo "  下一步: 运行环境检查"
echo "    python scripts/check_env.py"
echo "============================================================"
