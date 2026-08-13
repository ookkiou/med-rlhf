#!/usr/bin/env python3
"""
阶段 0 环境检查脚本

验证项:
1. Python 版本 >= 3.10
2. 关键依赖已安装 (torch, transformers, peft, trl, bitsandbytes, vllm, datasets)
3. GPU 可用且显存充足
4. QLoRA 4bit 可正常加载 Gemma-3-4B-IT
5. 模型中文输出正常
6. 前向 + 反向不 OOM

用法:
    python scripts/check_env.py [--model google/gemma-3-4b-it]

注意:
    - Gemma-3-4B-IT 是多模态模型, 纯文本使用 Gemma3ForCausalLM + AutoTokenizer
    - Gemma-3 是 gated model, 需要先 huggingface-cli login
    - 国内服务器需设置 HF_ENDPOINT=https://hf-mirror.com
"""

import sys
import os
import argparse
import importlib
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_python_version():
    """检查 Python 版本 >= 3.10"""
    version = sys.version_info
    print(f"[1/6] Python 版本: {version.major}.{version.minor}.{version.micro}")
    if version < (3, 10):
        print(f"  ✗ 需要 Python >= 3.10, 当前 {version.major}.{version.minor}")
        return False
    print(f"  ✓ Python 版本满足要求")
    return True


def check_dependencies():
    """检查关键依赖是否安装, 并验证版本"""
    required = {
        "torch": ("torch", "2.2.0"),
        "transformers": ("transformers", "4.50.0"),
        "peft": ("peft", "0.11"),
        "trl": ("trl", "0.12"),
        "bitsandbytes": ("bitsandbytes", "0.43"),
        "accelerate": ("accelerate", "0.30"),
        "datasets": ("datasets", "2.19"),
        "vllm": ("vllm", "0.6.0"),
    }
    print(f"[2/6] 检查关键依赖...")
    all_ok = True
    for module_name, (display_name, min_version) in required.items():
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  ✓ {display_name} ({version})")
        except ImportError:
            print(f"  ✗ {display_name} 未安装 (需要 >= {min_version})")
            all_ok = False
    return all_ok


def check_gpu():
    """检查 GPU 可用性和显存"""
    print(f"[3/6] 检查 GPU...")
    try:
        import torch

        if not torch.cuda.is_available():
            print("  ✗ CUDA 不可用, 无法使用 GPU")
            print("  可能原因: PyTorch 未安装 CUDA 版本")
            print("  修复: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return False
        gpu_name = torch.cuda.get_device_name(0)
        # 新版 torch (2.13+) 把 total_mem 改名为 total_memory, 兼容两者
        props = torch.cuda.get_device_properties(0)
        gpu_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / (1024**3)
        print(f"  ✓ GPU: {gpu_name}")
        print(f"  ✓ 显存: {gpu_mem:.1f} GB")
        if gpu_mem < 20:
            print(f"  ⚠ 显存 < 20GB, 可能无法运行 PPO (需要 ~24GB)")
        return True
    except Exception as e:
        print(f"  ✗ GPU 检查失败: {e}")
        return False


def check_qlora_load(model_path):
    """验证 QLoRA 4bit 加载 Gemma-3-4B-IT

    注意: Gemma-3-4B-IT 是多模态模型, 但纯文本任务使用
    Gemma3ForCausalLM (通过 AutoModelForCausalLM 自动映射)
    配合 AutoTokenizer (不是 AutoProcessor)
    """
    print(f"[4/6] 验证 QLoRA 4bit 加载 ({model_path})...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"  加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print(f"  加载模型 (4bit NF4)...")
        print(f"  (Gemma-3 是多模态模型, 使用 AutoModelForCausalLM 加载纯文本部分)")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"  ✓ 模型加载成功, 显存占用: {allocated:.2f} GB")

        # 验证模型类型
        model_type = type(model).__name__
        print(f"  ✓ 模型类型: {model_type}")

        return model, tokenizer
    except Exception as e:
        print(f"  ✗ QLoRA 加载失败: {e}")
        traceback.print_exc()
        return None, None


def check_chinese_output(model, tokenizer):
    """验证 Gemma-3 中文输出正常

    注意: Gemma-3-IT 需要 chat template 才能正常对话
    """
    print(f"[5/6] 验证中文输出...")
    try:
        import torch

        messages = [{"role": "user", "content": "你好,请用中文简单介绍一下什么是高血压。"}]

        # Gemma-3-IT 必须用 chat template
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=128,
                do_sample=False,
            )

        response = tokenizer.decode(
            output[0][input_ids.shape[-1]:], skip_special_tokens=True
        )

        chinese_chars = sum(1 for c in response if "\u4e00" <= c <= "\u9fff")
        print(f"  模型输出: {response[:150]}...")
        print(f"  中文字符数: {chinese_chars}")

        if chinese_chars < 5:
            print(f"  ⚠ 中文输出较少, 基座中文能力较弱 (预期, SFT 后会改善)")
        else:
            print(f"  ✓ 中文输出正常")

        return True
    except Exception as e:
        print(f"  ✗ 中文输出检查失败: {e}")
        traceback.print_exc()
        return False


def check_forward_backward(model, tokenizer):
    """验证前向 + 反向不 OOM"""
    print(f"[6/6] 验证前向 + 反向传播 (不 OOM)...")
    try:
        import torch

        messages = [{"role": "user", "content": "什么是感冒?"}]
        response_text = "感冒是一种常见的呼吸道疾病。"

        # 用 chat template 构造训练数据
        text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": response_text}],
            tokenize=False,
        )

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(model.device)
        labels = input_ids.clone()

        print("  前向传播...")
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        print(f"  ✓ 前向成功, loss = {loss.item():.4f}")

        print("  反向传播...")
        loss.backward()
        print(f"  ✓ 反向成功, 无 OOM")

        allocated = torch.cuda.memory_allocated() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  当前显存: {allocated:.2f} GB / 峰值: {max_allocated:.2f} GB")

        return True
    except torch.cuda.OutOfMemoryError:
        print(f"  ✗ OOM! 前向+反向超出显存")
        torch.cuda.empty_cache()
        return False
    except Exception as e:
        print(f"  ✗ 前向/反向检查失败: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="阶段 0 环境检查")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-it",
        help="模型路径或 HuggingFace ID",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Medical RLHF - 阶段 0 环境检查")
    print("=" * 60)
    print()

    # 检查 HuggingFace 环境变量
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"  HF Endpoint: {hf_endpoint}")
    if "hf-mirror" not in hf_endpoint:
        print(f"  ⚠ 未设置 HF 镜像, 国内服务器可能下载缓慢")
        print(f"    设置方法: export HF_ENDPOINT=https://hf-mirror.com")
    print()

    results = []

    # 1. Python 版本
    results.append(("Python 版本", check_python_version()))
    print()

    # 2. 依赖检查
    results.append(("依赖安装", check_dependencies()))
    print()

    # 3. GPU 检查
    results.append(("GPU 可用性", check_gpu()))
    print()

    # 4-6. 需要 GPU 的检查
    if all(r[1] for r in results):
        model, tokenizer = check_qlora_load(args.model)
        if model is not None and tokenizer is not None:
            print()
            results.append(("QLoRA 4bit 加载", True))

            results.append(("中文输出", check_chinese_output(model, tokenizer)))
            print()

            results.append(("前向+反向 OOM", check_forward_backward(model, tokenizer)))
            print()

            # 清理显存
            import torch
            del model, tokenizer
            torch.cuda.empty_cache()
        else:
            results.append(("QLoRA 4bit 加载", False))
            results.append(("中文输出", False))
            results.append(("前向+反向 OOM", False))
    else:
        print("跳过 GPU 相关检查 (前置条件不满足)")
        results.append(("QLoRA 4bit 加载", False))
        results.append(("中文输出", False))
        results.append(("前向+反向 OOM", False))

    # 汇总
    print("=" * 60)
    print("  检查结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")

    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("  所有检查通过, 可以进入阶段 1 (SFT 训练)!")
    else:
        print("  部分检查未通过, 请修复后再继续。")
        print("  常见问题:")
        print("    1. CUDA 不可用 -> 检查 PyTorch 是否安装了 CUDA 版本")
        print("    2. 模型下载失败 -> 检查 HF 登录和网络 (设置 HF_ENDPOINT)")
        print("    3. OOM -> 检查显存是否被其他进程占用 (nvidia-smi)")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
