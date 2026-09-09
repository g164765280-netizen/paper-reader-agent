# -*- coding: utf-8 -*-
"""Qwen3-VL 双时相变化检测测试：给 T1(A) + T2(B) 两期影像，问发生了什么变化。"""
import sys
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = "/media/sdb1/yjl/ollama-models/Qwen3-VL-8B-Instruct"
BASE = "/media/sdb1/gzj/data/LEVIR-CD/test"
scene = sys.argv[1] if len(sys.argv) > 1 else "test_103.png"

print("loading Qwen3-VL-8B ...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, local_files_only=True, trust_remote_code=False,
    torch_dtype=torch.bfloat16, device_map="auto")

img_a = Image.open(f"{BASE}/A/{scene}").convert("RGB")
img_b = Image.open(f"{BASE}/B/{scene}").convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image", "image": img_a},
    {"type": "image", "image": img_b},
    {"type": "text", "text": "这是同一区域不同时间拍摄的两张遥感影像（第一张是 T1，第二张是 T2）。请对比并说明两期之间发生了什么变化（如建筑新增/拆除、道路变化、植被变化等），并指出变化大致在影像的哪个位置。"},
]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=[text], images=[img_a, img_b], return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
ans = processor.decode(out[0], skip_special_tokens=True)
print("=== 回答 ===")
print(ans)
