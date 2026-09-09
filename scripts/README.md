# 复现与运行指南

## 服务器环境（4090）

| 路径 | 内容 |
|---|---|
| `/media/sdb1/gzj/data/rscd/corpus/` | 语料（全文 PDF + 解析 + 索引 + 评测集） |
| `/media/sdb1/gzj/weights/llm/rscd/` | bge-m3、bge-reranker、SFT/DPO/GRPO adapter、merged |
| `/media/sdb1/gzj/generated_data/llm/rscd/` | 所有评测结果 |
| `/media/sdb1/gzj/support/llm/rscd/` | 运行脚本（与本仓库 src/ 同步） |
| `/media/sdb1/yjl/ollama-models/` | Qwen3-8B、Qwen3-VL-8B（Transformers 格式） |

## 两个 Python 环境

| 环境 | 用途 | 关键库 |
|---|---|---|
| `all-in-rag` | RAG/评测/多模态/agent | transformers 4.57 + torchvision + sentence-transformers + faiss + mcp |
| `medicalgpt`（venv） | 训练（SFT/DPO/GRPO） | transformers 5.14 + peft + trl 1.9 |

## 复现步骤

```bash
# 1. 语料（arXiv 120 篇）
python src/corpus/fetch_abstracts.py --out data/corpus --target 120

# 2. 全文解析 + 建索引（句子级/LlamaIndex）
python src/rag/build_fulltext_index.py          # 段落级
python src/rag/build_llama_index.py             # LlamaIndex 切块（推荐）

# 3. 指标查找表（数字感知检索的关键）
python src/rag/build_metric_lookup.py

# 4. 出题 + 评测
python src/eval/eval_set_build_v2.py            # 生成 467 题
python src/eval/eval_run.py --configs rag --model Qwen3-8B

# 5. 训练（可选）
bash src/train/run_rscd_sft_v2.sh               # SFT 200 步
bash src/train/run_rscd_dpo.sh                  # DPO
python src/train/grpo_training_rscd.py ...      # GRPO（规则 verifier）

# 6. Agent（多模态 + 工具）
python src/agent/agent_demo.py                  # 路由式 agent demo
python src/agent/agent_toolcall.py "问题"        # 自主 tool-calling

# 7. MCP server（工具协议化）
python src/agent/mcp_server.py                  # 暴露 6 个工具
```

## 关键模型下载（走 hf-mirror，避开墙）

```bash
HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='./bge-m3')"
# bge-reranker-v2-m3、Qwen3-8B、Qwen3-VL-8B 同理
```

## 注意

- 磁盘 `/media/sdb1` 曾 99% 满导致极慢（~10MB/s），模型/语料建议放快盘。
- arXiv 下载有 429 限流，脚本已带退避；批量下载建议从本机下再 scp。
- Qwen3-8B 默认带 `<think>` 推理块，RAG 生成时可用 prompt 抑制或后处理剥离。
