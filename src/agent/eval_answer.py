# -*- coding: utf-8 -*-
"""数字题评测：本地 Qwen3-8B 读题+查库答题，与 gold 比对。

- 每题：喂"该论文的指标记录(方法×数据集×指标→数值) + 题目"，Qwen3-8B 答数值。
- 打分：Qwen 答的数值 == gold 值？
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CONF = "EVAL"
QFILE = Path("/media/sdb1/gzj/data/rscd/corpus/eval_set_v9.jsonl")
LIB = Path("/media/sdb1/gzj/data/rscd/corpus/metric_library.jsonl")
MODEL_PATH = "/media/sdb1/yjl/ollama-models/Qwen3-8B"


def load_lib() -> dict[str, list[dict]]:
    by_paper: dict[str, list[dict]] = {}
    for l in LIB.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            by_paper.setdefault(r["paper_id"], []).append(r)
    return by_paper


def fmt_context(recs: list[dict]) -> str:
    lines = []
    for r in recs:
        lines.append(f"- {r['method']} / {r['dataset']} / {r['metric']} / {r['value']}")
    return "\n".join(lines)


def nums(text: str):
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]


def match(gold: str, ans: str) -> bool:
    g = float(gold)
    for a in nums(ans):
        if abs(a - g) <= max(0.01, g * 0.02):
            return True
    return False


def main():
    qs = [json.loads(l) for l in QFILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    lib = load_lib()

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    results = []
    correct = 0
    for i, q in enumerate(qs, 1):
        recs = lib.get(q["paper_id"], [])
        ctx = fmt_context(recs)
        prompt = ("你是遥感论文问答助手。下面给出一篇论文的实验结果数据，以及一个关于该论文的问题。\n"
                  "题目里说的\"提出的方法\"就是该论文自己提出的方法（通常是 Ours/Proposed 那一行，"
                  "或方法名和论文标题缩写一致的那行）。严格根据数据回答，只输出数值，不要解释。\n\n"
                  f"论文实验结果数据：\n{ctx}\n\n问题：{q['question']}\n\n答案（只写数值）：")
        msgs = [{"role": "system", "content": "你是问答助手，直接输出答案数字，不要思考、不要解释。"},
                {"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, enable_thinking=False, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(ids, max_new_tokens=60, do_sample=False)
        raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        # 兜底：若有 " response" 标记则取其后（关思考后通常没有）
        ans = raw.split(" response")[-1].strip() if " response" in raw else raw
        ok = match(q["gold"]["value"], ans)
        correct += ok
        results.append({**q, "model_ans": ans, "correct": ok})
        print(f"[{i}/{len(qs)}] 累计正确 {correct}/{i}", flush=True)

    # 存结果
    out = Path("/media/sdb1/gzj/data/rscd/corpus/eval_result_v9.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    print(f"\n最终答对率: {correct}/{len(qs)} = {correct/len(qs)*100:.1f}%")


if __name__ == "__main__":
    main()