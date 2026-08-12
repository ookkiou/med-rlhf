"""
SFT 数据清洗 + 抽样

从 shibing624/medical (240 万条) 中按比例抽样 5-8 万条，
构造 sft_medical.jsonl 供 SFTTrainer 使用。

抽样策略:
  - 医疗问答（中文化）   50%  ~30,000
  - 医疗对话（多轮）     25%  ~15,000
  - 医学教材/百科 QA     15%   ~9,000
  - 安全合规（拒答/引导就医）10% ~6,000

TODO:
  1. 加载 shibing624/medical 数据集
  2. 按类别分组 + 按比例抽样
  3. 适配 Gemma-3 chat template
  4. hash 对比过滤 CMB/CMExam 测试题（防数据污染）
  5. 输出 data/sft_medical.jsonl
"""
