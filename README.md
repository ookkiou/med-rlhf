# med-rlhf-ppo

> 基于 RLHF 的中文医疗垂域大模型后训练：SFT 注入领域知识 → DPO 离线偏好对齐（保底） → PPO 在线强化学习（冲刺）。


## 项目简介

在 **Gemma-3-4B-IT** 上完成中文医疗垂域后训练全流程，分两步走：

1. **DPO 保底**：SFT 注入领域知识 → DPO 离线对齐医疗安全偏好 → 产出「基座 / +SFT / +DPO」三档对比
2. **PPO 冲刺**：在 SFT checkpoint 上训练独立 RM → PPO 四模型在线 rollout → 产出第四档「+SFT+PPO」

最终交付取决于 PPO 是否跑通：跑通写 PPO，放弃则 DPO 保底。在 CMB 和 CMExam 两个公开医疗 benchmark 上给出最多四档对比数据，并配合 LLM-judge 做主观安全性评测。

## 技术栈

- **模型**：Gemma-3-4B-IT（Google 架构，QLoRA 4bit NF4 量化，单卡 24GB）
- **训练**：HuggingFace Transformers + PEFT + TRL（SFTTrainer / DPOTrainer / RewardTrainer / PPOTrainer）
- **推理**：vLLM
- **评测**：CMB（综合医疗 benchmark）、CMExam（中国国家医师考试题）、LLM-judge 安全性打分
- **数据**：shibing624/medical（240 万条中文医疗数据）、Medical-ChatBot-DPO（偏好对）、自建安全偏好对（2,000 条高风险场景）

## 项目结构

```
med-rlhf-ppo/
├── configs/                        # 训练超参配置（sft / dpo / rm / ppo）
├── data/
│   ├── prepare_medical_data.py     # SFT 数据清洗 + 抽样
│   └── build_safety_preference.py  # 自建安全偏好对
├── scripts/                        # 训练/评测启动脚本
├── evaluate/
│   ├── evaluate_medical.py         # CMB/CMExam 客观评测
│   └── llm_judge_safety.py         # 安全性三维度 LLM 打分
├── results/                        # 评测结果与对比表
├── Gemma-3-4B医疗垂域后训练方案.md  # 完整方案文档
└── requirements.txt
```

## 流程

```
Gemma-3-4B-IT
    │
    ├─ 阶段 1: QLoRA SFT（医疗指令数据 5–8 万条）
    │   └─产出→ SFT checkpoint
    │
    ├─ 阶段 2: DPO（医疗偏好对 2–3 万条）
    │   ├─ policy（SFT checkpoint，训练）
    │   └─ ref（SFT checkpoint，冻结，KL 约束）
    │   └─产出→ DPO checkpoint       ← 【保底交付点】
    │
    ├─ 阶段 3: 三档评测（基座 / +SFT / +DPO）
    │   ├─ 客观：CMB / CMExam
    │   └─ 主观：LLM-judge 安全性
    │   └─产出→ 三档对比表            ← 【保底交付物】
    │
    ┊─ 阶段 4（可选 · PPO 冲刺）──────────────────────────
    │   ├─ 训练独立 RM（复用 DPO 偏好数据）
    │   └─ PPO 500 步（actor/critic/ref/reward 四模型协作）
    │       └─产出→ PPO checkpoint + 第四档评测
    │
    └─ 最终交付：四档对比表（PPO 跑通）或 三档对比表（放弃 PPO）
```

## 结果（预期，实测替换）

| 模型 | CMB | CMExam | 安全性(LLM-judge) | 引导就医率 |
|---|---|---|---|---|
| 基座 Gemma-3-4B-IT | ~38 | ~45 | ~3.0 / 5 | ~15% |
| +SFT | ~62 | ~70 | ~3.4 / 5 | ~30% |
| +SFT+DPO（保底） | ~64 | ~72 | **~4.3 / 5** | **~65%** |
| +SFT+PPO（冲刺） | ~66 | ~74 | ~4.5 / 5 | ~70% |

> 核心消融结论：SFT 贡献客观指标主要增益；DPO 贡献主观安全性主要增益；PPO 在 DPO 基础上小幅再抬一档（在线探索的边际增益）。
> 详见 [results/comparison.md](results/comparison.md)。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备 SFT 数据
python data/prepare_medical_data.py

# 3. SFT（必做）
bash scripts/train_sft.sh

# 4. 构造安全偏好对
python data/build_safety_preference.py

# 5. DPO（必做）
bash scripts/train_dpo.sh

# 6. 三档评测（必做 · 保底交付）
bash scripts/evaluate.sh

# ===== 以下为可选 PPO 冲刺 =====

# 7. 训练奖励模型 RM
bash scripts/train_rm.sh

# 8. PPO
bash scripts/train_ppo.sh

# 9. 第四档评测
bash scripts/evaluate_ppo.sh
```

## 致谢

- [MedicalGPT](https://github.com/shibing624/MedicalGPT) - 训练流程设计参考
- [shibing624/medical](https://huggingface.co/datasets/shibing624/medical) - 医疗训练数据
- [CMB](https://github.com/FreedomIntelligence/CMB) - 中文医疗评测基准
- [HuggingFace TRL](https://huggingface.co/docs/trl) - 训练框架
- [Gemma-3](https://ai.google.dev/gemma) - 底座模型
