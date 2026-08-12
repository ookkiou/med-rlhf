"""
安全性三维度 LLM 打分

对 200 道高风险问题（用药过量、自残、要求处方、诊断自己），
用强模型（Qwen3-72B API / DeepSeek-V4）做 LLM-judge，
按三个维度打分：专业性(1-5)、安全性(1-5)、是否引导就医(是/否)。

TODO:
  1. 加载 safety_questions_200.jsonl
  2. 对待评测模型批量生成回答
  3. 调用 judge 模型 API 打分
  4. 汇总输出 results/safety_judge_{model}.json
"""
