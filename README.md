# 领域文献速读 Agent（PaperReader）

> 面向**研究生新生**：对自己领域还不熟悉时，快速读懂文献、查指标、看影像、写实验报告。
> 同时是一个**可学习、可写进简历**的 Agent 开发完整案例（RAG → 微调 → RL → 多模态 → 工具/MCP）。
> 示例领域：**遥感变化检测（RS-CD）**。

---

## 一、这个项目解决什么问题

研究生新生刚进组，最痛的是：**领域不熟、论文读不懂、术语看不懂、实验不会分析**。

这个 Agent 帮你：

| 能力 | 例子 | 结果 |
|---|---|---|
| **论文问答** | "LEVIR-CD 训练样本多少？" | 带引用的准确回答 |
| **查指标** | "这篇论文 F1 多少？" | 直接报数字（数字忠实 97.2%） |
| **看影像** | "这张遥感影像有什么？" | 准确描述地表类型 |
| **测变化** | "T1/T2 哪里变了？" | 变化检出 100%、定位 85% |
| **写实验报告** | 给变化图 + 指标 | 自动生成规范分析报告 |
| **多轮对话** | "上一条那个 F1 是哪篇论文？" | 有记忆的追问 |
| **自主工具调用** | 复杂问题 | 自动决定调哪个工具 |

## 二、技术栈（简历关键词）

```
RAG 检索增强 · SFT/DPO/GRPO 微调对齐 · Qwen3-8B/Qwen3-VL 多模态
FAISS + BM25 混合检索 · bge-m3 + rerank · 数字感知检索 · tool-calling · MCP
```

## 三、核心结果（有数据支撑）

| 指标 | 值 |
|---|---|
| 检索召回（论文级） | 99.6% |
| 数字忠实 | **97.2%** |
| 名称/数据集覆盖 | 77.3% |
| 引用率 | 98.2% |
| 拒答（不瞎编） | 100% |
| 变化检出（多模态） | 100% |
| 变化定位（多模态） | 85% |

## 四、目录结构

```
领域文献阅读Agent/
├── README.md             # 本文件
├── docs/
│   ├── 项目报告.md        # 完整实验报告（数据+结论）
│   ├── 学习路线.md        # 从 0 到 1 学 agent 开发的路线
│   └── 简历与面试.md      # 简历 bullets + 面试讲解
├── src/
│   ├── corpus/           # 语料抓取（arXiv 论文）
│   ├── rag/              # RAG 管线（检索/索引/生成）
│   ├── eval/             # 评测（Oracle 对照 + 多模态评测）
│   ├── train/            # 训练（SFT/DPO/GRPO）
│   ├── agent/            # Agent + 工具 + MCP
│   └── vlm/              # 多模态（Qwen3-VL）
├── data/                 # 训练数据、评测集
└── scripts/              # 一键运行脚本
```

## 五、快速开始

> 需要：2×RTX 4090（或 1 张 24G 卡）、Python 环境、Qwen3-8B / Qwen3-VL-8B / bge-m3 / bge-reranker。

```bash
# 1. 抓语料（arXiv 论文）
python src/corpus/fetch_abstracts.py --out data/corpus

# 2. 解析全文 + 建索引
python src/rag/build_fulltext_index.py

# 3. 出题 + 评测
python src/eval/eval_set_build_v2.py
python src/eval/eval_run.py --configs rag

# 4. 启动 Agent（多模态 + 工具）
python src/agent/agent_demo.py

# 5. 启动 MCP server（工具协议化）
python src/agent/mcp_server.py
```

## 六、两条学习主线

1. **读文献主线**：把语料换成你自己的领域论文，就能得到你的领域文献速读 Agent。
2. **学 Agent 开发主线**：按 `docs/学习路线.md` 的步骤，从 RAG 一路做到 tool-calling + MCP + 多模态。

---

*本项目为教学/工程演示，输出内容不构成专业建议。*
