# -*- coding: utf-8 -*-
"""Qwen3-VL 冒烟测试：加载 + 描述一张遥感影像。"""
import sys
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = "/media/sdb1/yjl/ollama-models/Qwen3-VL-8B-Instruct"
IMG = sys.argv[1] if len(sys.argv) > 1 else "/media/sdb1/gzj/data/LEVIR-CD/test/A/test_103.png"

print("loading Qwen3-VL-8B ...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, local_files_only=True, trust_remote_code=False,
    torch_dtype=torch.bfloat16, device_map="auto")

img = Image.open(IMG).convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image", "image": img},
    {"type": "text", "text": "请描述这张遥感影像里有什么，属于什么地表类型（如建筑、农田、水体、道路等）。"},
]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
ans = processor.decode(out[0], skip_special_tokens=True)
print("=== 回答 ===")
print(ans)
