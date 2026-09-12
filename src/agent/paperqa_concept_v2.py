# -*- coding: utf-8 -*-
"""概念题 PaperQA2 通路 v2：本地 Qwen3-8B(vLLM) + 本地 bge-m3 embedding。

相比 v1 的改进（针对 6 道错题的 3 类失败模式，全部落在 RAG/prompt 层，模型仍是本地 8B）：
1. 同名实体污染（cq12 SECOND）：
   - qa/system prompt 强化"歧义实体先消歧，取遥感变化检测语境"
   - 提高 evidence_relevance_score_cutoff=3，过滤弱相关 chunk
2. 引用错位拼接（cq14 配准误差）：
   - 提高 cutoff 过滤消融实验类干扰证据
   - summary prompt 强调"实验消融数据除非直接相关否则判 not applicable"
3. 术语不精（cq03/06/09/13）：
   - qa prompt 强调"先给出该概念的标准定义/本质特征，再展开"

用法：
  /media/sdb1/gzj/support/llm/envs/paperqa/bin/python paperqa_concept_v2.py
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
# 提高证据相关度阈值，过滤弱相关/消融实验类干扰（cq14 根因）
SETTINGS.answer.evidence_relevance_score_cutoff = 3
# 证据数（保留默认 10）
SETTINGS.answer.evidence_k = 10
# 覆盖 prompt
SETTINGS.prompts.qa = QA_PROMPT
SETTINGS.prompts.system = QA_SYSTEM
SETTINGS.prompts.summary = SUMMARY_PROMPT

FULLTEXT_DIR = Path("/media/sdb1/gzj/data/rscd/corpus/fulltext")
QUESTIONS = Path("/media/sdb1/gzj/support/llm/rscd/concept_questions.jsonl")
OUT = Path("/media/sdb1/gzj/support/llm/rscd/concept_answers_v2.jsonl")
TXT_DIR = Path("/media/sdb1/gzj/support/llm/rscd/fulltext_txt")

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = 全量论文


def fulltext_to_json(pdf_json: Path) -> str:
    d = json.loads(pdf_json.read_text(encoding="utf-8"))
    pt = d.get("pages_text", [])
    if isinstance(pt, list):
        return "\n".join(str(p) for p in pt)
    return str(pt)


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