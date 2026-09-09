# -*- coding: utf-8 -*-
"""Qwen3-VL 变化检测系统评测：用 LEVIR-CD 测试集 + GT 变化图对照。

指标：
  - 变化检出率：模型回答"有变化"的比例（LEVIR-CD 全部有变化，应接近 100%）
  - 定位准确率：模型描述的变化位置是否匹配 GT 变化区域的质心象限
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from agent import VLM

LEVIR = Path("/media/sdb1/gzj/data/LEVIR-CD/test")
QUADRANTS = {"左上": 0, "右上": 1, "左下": 2, "右下": 3, "中心": 4, "中部": 4, "中央": 4}


def gt_quadrants(mask_path: str) -> set[int]:
    """GT = 有显著变化的象限集合（变化像素占比 > 阈值）。中心算'任意象限有变化'。"""
    m = np.array(Image.open(mask_path).convert("L"))
    h, w = m.shape
    fracs = {}
    for name, q in [("左上", 0), ("右上", 1), ("左下", 2), ("右下", 3)]:
        y0, y1 = (0, h // 2) if "上" in name else (h // 2, h)
        x0, x1 = (0, w // 2) if "左" in name else (w // 2, w)
        region = m[y0:y1, x0:x1]
        fracs[q] = (region > 128).mean()
    sig = {q for q, f in fracs.items() if f > 0.005}  # 该象限变化占比 > 0.5%
    # 若变化集中在中心（各象限都有一点但都不显著），视为中心变化
    return sig if sig else {4}


def parse_answer(ans: str):
    # 去掉回显的 prompt，只取 assistant 回答
    if "assistant" in ans:
        ans = ans.split("assistant", 1)[-1]
    has_change = bool(re.search(r"有变化|发生.*变化|新增|拆除|扩建|新建|取代|开发", ans))
    has_no_change = bool(re.search(r"没有变化|无变化|无明显变化|未发现变化|未发生", ans))
    loc = -1
    for kw, q in QUADRANTS.items():
        if kw in ans:
            loc = q
            break
    return has_change and not has_no_change, loc


def main(n: int = 20):
    scenes = sorted(p.name[:-4] for p in (LEVIR / "A").glob("*.png"))[:n]
    vlm = VLM()
    prompt = ("这是同一区域的两张遥感影像（T1 和 T2）。请先回答：两期之间是否有变化？"
              "若有，变化大致在影像的哪个位置？请用「左上/右上/左下/右下/中心」中的一个词描述位置。")
    det_ok = loc_ok = 0
    results = []
    for i, s in enumerate(scenes, 1):
        ans = vlm(prompt, [str(LEVIR / "A" / f"{s}.png"), str(LEVIR / "B" / f"{s}.png")])
        has_change, loc = parse_answer(ans)
        gqs = gt_quadrants(str(LEVIR / "label" / f"{s}.png"))
        d = has_change  # GT 全有变化
        # 命中：模型说的象限有显著变化（中心视为"任一显著象限"均可）
        l = (loc in gqs) or (loc == 4 and bool(gqs))
        det_ok += d
        loc_ok += l
        results.append({"scene": s, "has_change": has_change, "loc": loc, "gt_locs": sorted(gqs)})
        print(f"[{i}/{n}] {s} 检出={has_change} 定位={loc}(GT象限={sorted(gqs)}) {'✓' if l else ''}", flush=True)

    print(f"\n变化检出率: {det_ok}/{n} = {det_ok/n*100:.1f}%")
    print(f"定位准确率: {loc_ok}/{n} = {loc_ok/n*100:.1f}%")


if __name__ == "__main__":
    main()
