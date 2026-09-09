# -*- coding: utf-8 -*-
"""变化图 → 自动实验报告 demo。

输入：GT 变化图（mask）+ 指标数字 + 方法/数据集名
流程：变化图上色（红=变化）→ Qwen3-VL 看变化图（空间）→ 融合指标（定量）→ 完整报告
"""
import sys
from PIL import Image
from agent import VLM

LEVIR = "/media/sdb1/gzj/data/LEVIR-CD/test"


def colorize(mask_path: str, out="/tmp/change_map_color.png") -> str:
    m = Image.open(mask_path).convert("L")
    # 红=变化(白)，背景用浅灰
    rgb = Image.new("RGB", m.size, (235, 235, 235))
    px = rgb.load()
    mp = m.load()
    for y in range(m.size[1]):
        for x in range(m.size[0]):
            if mp[x, y] > 128:
                px[x, y] = (255, 60, 60)  # 红色变化区域
    rgb.save(out)
    return out


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "test_103"
    method = "ChangeFormer"
    dataset = "LEVIR-CD"
    metrics = "F1=0.892, IoU=0.805, OA=0.990, Precision=0.910, Recall=0.874"

    mask = f"{LEVIR}/label/{scene}.png"
    cm = colorize(mask)

    vlm = VLM()
    prompt = (
        f"方法 {method} 在 {dataset} 数据集上，检测指标为 {metrics}。"
        "下面这张图里，红色区域是模型检测出的变化区域。"
        "请写一段规范的结果分析报告，包含：总体结论、指标解读（Precision/Recall 权衡）、"
        "空间变化分布描述、误差诊断、改进建议。"
    )
    print("=== 实验报告 ===")
    print(vlm(prompt, [cm]))


if __name__ == "__main__":
    main()
