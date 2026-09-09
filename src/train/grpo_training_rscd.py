# -*- coding: utf-8 -*-
"""RS-CD 的 GRPO 训练：规则 verifier（引用 + 数字忠实），复用 MedicalGPT 的 GRPO 结构。

数据集：rscd_sft_v2.jsonl → {"prompt": "证据[1]...\n\n问题...", "answer": "gold"}
reward：
  - citation_reward：回答里出现 [n] 引用 → +1
  - number_reward：回答里的数字与 gold 数字一致 → +1
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------------------
# 规则 verifier
# ---------------------------------------------------------------------------
def citation_reward(completions, **kwargs):
    # TRL 1.9.x: completions 是字符串列表
    return [1.0 if re.search(r"\[\d+\]", c) else 0.0 for c in completions]


def number_reward(completions, answer, **kwargs):
    rewards = []
    for content, gold in zip(completions, answer):
        gold_nums = re.findall(r"\d+\.\d+", str(gold))
        content_nums = [float(x) for x in re.findall(r"\d+\.\d+", content)]
        r = 0.0
        for g in gold_nums:
            gv = float(g)
            if any(abs(gv - c) <= max(0.001, abs(gv) * 0.02) for c in content_nums):
                r = 1.0
                break
        rewards.append(r)
    return rewards


def load_grpo_dataset(sft_path: Path, limit: int = 800) -> Dataset:
    rows = []
    for line in sft_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        conv = d["conversations"]
        human = next(m["value"] for m in conv if m["from"] == "human")
        gpt = next(m["value"] for m in conv if m["from"] == "gpt")
        rows.append({"prompt": human, "answer": gpt})
        if len(rows) >= limit:
            break
    return Dataset.from_list(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--limit", type=int, default=800)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        p.data = p.data.to(torch.float32)

    ds = load_grpo_dataset(Path(args.data), args.limit)

    grpo_config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=128,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        logging_steps=1,
        bf16=True,
        report_to="none",
        save_steps=args.steps,
        save_total_limit=1,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[citation_reward, number_reward],
        args=grpo_config,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("GRPO_DONE", args.output)


if __name__ == "__main__":
    main()
