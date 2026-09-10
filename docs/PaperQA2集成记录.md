## PaperQA2 证据层集成记录（2026-09-10）

**已打通**：
- LLM：远程 qwen3.8-flash（tokenrhythm，环境变量 OPENAI_API_KEY/OPENAI_API_BASE）
- embedding：本地 bge-m3（st- 前缀，sentence-transformers）
- 跳过 S2/Crossref 元数据抓取：Settings.parsing.use_doc_details=False

**4 个集成坑**：
1. LMI 不读 litellm_params 里的 api_key → 用环境变量
2. aadd 也要传 settings（默认用 gpt-4o 报"模型不可用"）
3. S2 429 限流 → use_doc_details=False
4. pillow 缺失 → pip install pillow

**当前状态**：3 篇论文 aadd+aquery 跑通，回答"insufficient information"（3 篇太少，属诚实行为）。全量 120 篇需 ~40 分钟 embedding。
