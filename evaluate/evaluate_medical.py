"""
CMB / CMExam 客观评测

对基座 / SFT / DPO / PPO checkpoint 在 CMB 和 CMExam 两个公开医疗 benchmark 上评测，
输出各档分数（选择题准确率）。

TODO:
  1. 加载 benchmark 数据集
  2. vLLM 批量推理
  3. 答案提取（选项字母 / 医学术语匹配）
  4. 计算准确率，输出 results/{model}_{benchmark}.json
"""
