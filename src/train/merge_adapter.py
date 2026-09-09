# -*- coding: utf-8 -*-
"""合并 LoRA adapter 到基座，输出可独立部署的完整模型。"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, max_shard_size="5GB")
    tok.save_pretrained(args.out)
    print(f"MERGED -> {args.out}")


if __name__ == "__main__":
    main()
