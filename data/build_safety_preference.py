"""
自建安全偏好对构造

从 SFT 数据中抽取 2,000 条高风险问题（自残、用药过量、诊断自己、要求处方），
用 SFT checkpoint 采样多个回答，标注 chosen（安全/引导就医）和 rejected（给处方/过度诊断），
产出 dpo_medical.jsonl 供 DPOTrainer / RewardTrainer 共用。

TODO:
  1. 从 sft_medical.jsonl 中筛选高风险问题
  2. 加载 SFT checkpoint，每题采样 4 个回答（vLLM）
  3. 人工/强模型标注 chosen & rejected
  4. 输出 data/dpo_medical.jsonl
"""
