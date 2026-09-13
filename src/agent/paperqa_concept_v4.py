# -*- coding: utf-8 -*-
"""概念题 PaperQA2 通路 v4：本地 Qwen3-8B(vLLM) + 权威概念锚点 + 关摘要直读原文。

在 v3 基础上（权威词条注入），关键改动：**关掉 summary 摘要环节**，让答题模型
直接读检索到的原文 chunk，跳过"8B 摘要转述时把 Precision/Recall、漏检/误报写反"
这个案发现场（cq39/cq40 的根因）。

- evidence_skip_summary=True：不调摘要 LLM，原文直接作为 context
- evidence_k 10→5、answer_max_sources 收紧：原文 chunk 长，减少召回数避免超上下文

用法：
  /media/sdb1/gzj/support/llm/envs/paperqa/bin/python paperqa_concept_v4.py
"""
from __future__ import annotations

import asyncio
import json
import glob
import sys
from pathlib import Path

from paperqa import Docs, Settings

VLLM_BASE = "http://localhost:8000/v1"
MODEL = "Qwen3-8B"

LLM_CONFIG = {
    "model_list": [
        {
            "model_name": MODEL,
            "litellm_params": {
                "model": f"hosted_vllm/{MODEL}",
                "api_base": VLLM_BASE,
                "api_key": "sk-no-key-required",
                "temperature": 0.1,
                "max_tokens": 1024,
                # 关闭 Qwen3 内建思考，避免输出被 thinking 污染
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        }
    ]
}

QA_PROMPT = """请根据下面的出处语境，回答这个遥感变化检测概念题。

【题目】
{question}

【出处语境】
{context}

【回答要求】
1. 先直接给出这个概念的本质定义/标准答案（一句话说清核心），再展开解释。
2. 若题目涉及同名或易歧义的实体（如 BIT、SECOND、ChangeFormer 等），请先明确它在本语境（遥感变化检测）下的具体含义，不要与其它领域（如 3D 点云目标检测、自动驾驶）的同名实体混淆。
3. 只回答概念本质，不要堆砌无关的实验细节或与问题无关的方法名称。
4. 每个结论后标注支持它的出处标记（形如 (pqac-xxxx)），只引用上方语境里出现过的标记。
5. 若出语境信息不足，直接回答"我无法根据现有语料回答"。

## 引用格式（只用逗号/空格分隔的括号标记）：
- 正确：(pqac-d79ef6fa, pqac-0f650d59)
## 错误示例（不要这样写）：
- (pqac-d79ef6fa and pqac-0f650d59)  (author year)  pqac-d79ef6fa 不加括号

回答（{answer_length}）："""

QA_SYSTEM = """你是遥感变化检测领域的专家，回答概念题。要求：
- 直接、简洁、专业。
- 对于歧义或缩写词，先明确它在遥感变化检测语境下的定义。
- 严格只用给出的出处语境里的信息作答，不凭记忆补充出处里没有的内容。
- 信息不足时诚实说明，不瞎编。"""

SUMMARY_PROMPT = """总结下面的摘录，以帮助回答一个关于遥感变化检测的问题。

摘录来自 {citation}
---
{text}
---
问题：{question}

不要直接回答问题，而是总结出能帮助回答问题的证据。保持详细；报告具体的数字、公式或直接引文（用引号标注）。如果摘录与问题【无关】，或只是实验消融/性能对比表格、与概念本身无直接关系，请回复"Not applicable"。最后用一行给出该摘录与问题的相关度整数分数（1-10），并解释分数。

相关证据总结（{summary_length}）："""

SETTINGS = Settings(
    llm=MODEL,
    llm_config=LLM_CONFIG,
    summary_llm=MODEL,
    summary_llm_config=LLM_CONFIG,
    embedding="st-/media/sdb1/gzj/weights/llm/rscd/bge-m3",
)
SETTINGS.parsing.use_doc_details = False
SETTINGS.agent.agent_llm = MODEL
SETTINGS.agent.agent_llm_config = LLM_CONFIG

# —— 关键调参（针对失败模式的 RAG 层修复）——
# 关摘要：原文直接当 context，跳过"8B 摘要写反术语"环节
SETTINGS.answer.evidence_skip_summary = True
# 关摘要后 relevance_score 默认 5（不走打分），cutoff 无意义，但保留无害
SETTINGS.answer.evidence_relevance_score_cutoff = 0
# 原文 chunk 长，把召回数压到 2，避免超 8K 上下文（关摘要后每个 chunk 是原文 5000 字符）
SETTINGS.answer.evidence_k = 2
SETTINGS.answer.answer_max_sources = 3
# 覆盖 prompt
SETTINGS.prompts.qa = QA_PROMPT
SETTINGS.prompts.system = QA_SYSTEM
SETTINGS.prompts.summary = SUMMARY_PROMPT

FULLTEXT_DIR = Path("/media/sdb1/gzj/data/rscd/corpus/fulltext")
QUESTIONS = Path("/media/sdb1/gzj/support/llm/rscd/concept_questions.jsonl")
OUT = Path("/media/sdb1/gzj/support/llm/rscd/concept_answers_v4.jsonl")
TXT_DIR = Path("/media/sdb1/gzj/support/llm/rscd/fulltext_txt")
# 权威概念词条（本地仓库 concepts.json）
CONCEPTS_FILE = Path("/media/sdb1/gzj/support/llm/rscd/concepts_authoritative.json")
# 待查证 / 弱来源的词条 id（不注入）
EXCLUDE_IDS = {"open_vocabulary", "changemamba", "sam_cd", "tinycd", "stanet", "changestar"}

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = 全量论文


def fulltext_to_json(pdf_json: Path) -> str:
    d = json.loads(pdf_json.read_text(encoding="utf-8"))
    pt = d.get("pages_text", [])
    if isinstance(pt, list):
        return "\n".join(str(p) for p in pt)
    return str(pt)


def build_authoritative_text() -> str:
    """把 concepts.json 里符合标准的词条拼成一份权威概念库文本。"""
    d = json.loads(CONCEPTS_FILE.read_text(encoding="utf-8"))
    concepts = d.get("concepts", [])
    lines = ["# 遥感变化检测领域权威概念库（人工整理）", ""]
    for c in concepts:
        if c.get("id") in EXCLUDE_IDS:
            continue
        zh = c.get("term_zh", "")
        en = c.get("term", "")
        definition = c.get("definition", "")
        explanation = c.get("explanation", "")
        # 术语行：中文名(英文名)：定义
        lines.append(f"## {zh}({en})")
        lines.append(f"定义：{definition}")
        if explanation:
            lines.append(f"说明：{explanation}")
        lines.append("")
    return "\n".join(lines)


async def main():
    qs = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"概念题 {len(qs)} 道", flush=True)

    fts = sorted(FULLTEXT_DIR.glob("*.json"))
    if LIMIT:
        fts = fts[:LIMIT]
    print(f"论文 {len(fts)} 篇", flush=True)

    docs = Docs()
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    added = 0
    for i, f in enumerate(fts, 1):
        d = json.loads(f.read_text(encoding="utf-8"))
        paper_id = d.get("arxiv_id", f.stem)
        title = " ".join((d.get("title", "unknown") or "unknown").split())[:120]
        text = fulltext_to_json(f)
        txt_path = TXT_DIR / f"{paper_id}.txt"
        try:
            txt_path.write_text(text, encoding="utf-8")
            await docs.aadd(str(txt_path), settings=SETTINGS, citation=paper_id,
                            title=title, docname=paper_id)
            added += 1
            print(f"[{i}/{len(fts)}] ok  {paper_id}  {title[:40]}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(fts)}] ERR {paper_id}: {str(e)[:160]}", flush=True)
    print(f"\naadd 完成：{added}/{len(fts)} 篇。语料文本数={len(docs.texts)}", flush=True)

    # —— 额外 aadd 权威概念库文档（不动原流程，只是多喂一份权威知识）——
    concepts_text = build_authoritative_text()
    concepts_path = TXT_DIR / "authoritative_concepts.txt"
    concepts_path.write_text(concepts_text, encoding="utf-8")
    n_concepts = concepts_text.count("## ")
    await docs.aadd(str(concepts_path), settings=SETTINGS,
                    citation="authoritative_concepts", title="遥感变化检测权威概念库",
                    docname="authoritative_concepts")
    print(f"权威概念库已注入：{n_concepts} 条词条。语料文本总数={len(docs.texts)}", flush=True)

    results = []
    for qi, q in enumerate(qs, 1):
        print(f"\n=== [{qi}/{len(qs)}] 提问：{q['question']} ===", flush=True)
        try:
            s = await docs.aquery(q["question"], settings=SETTINGS)
            ans = getattr(s, "answer", None)
            ans_str = str(ans) if ans is not None else ""
            contexts = []
            try:
                for ctx in getattr(s, "contexts", []):
                    t = getattr(ctx, "text", None)
                    doc = getattr(t, "doc", None)
                    contexts.append({
                        "id": getattr(ctx, "id", ""),
                        "citation": getattr(doc, "citation", "") if doc else "",
                        "docname": getattr(doc, "docname", "") if doc else "",
                        "name": getattr(t, "name", "") if t else "",
                        "context": getattr(ctx, "context", ""),
                        "score": getattr(ctx, "score", None),
                    })
            except Exception:
                contexts = []
            results.append({
                "id": q["id"],
                "question": q["question"],
                "key_points": q["key_points"],
                "answer": ans_str,
                "contexts": contexts,
                "references": getattr(s, "references", ""),
                "formatted_answer": getattr(s, "formatted_answer", None),
            })
            print("  回答:", (ans_str[:250] + "...") if len(ans_str) > 250 else ans_str, flush=True)
            print("  出处条数:", len(contexts), flush=True)
        except Exception as e:
            print(f"  aquery ERROR: {str(e)[:200]}", flush=True)
            results.append({
                "id": q["id"], "question": q["question"], "key_points": q["key_points"],
                "answer": "", "contexts": [], "error": str(e)[:200],
            })

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results),
                   encoding="utf-8")
    print(f"\n已保存 {len(results)} 条回答到 {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())