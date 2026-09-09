# -*- coding: utf-8 -*-
"""领域文献阅读 Agent 网页前端（Gradio）。

三个 Tab：
  1. 论文问答（多轮对话，走 tool-calling agent）
  2. 影像分析（单张=描述 / 两张=变化检测）
  3. 实验报告（变化图 + 指标 → 报告）
"""
from __future__ import annotations

import gradio as gr

from agent_toolcall import ToolCallAgent

print("加载模型（Qwen3-8B + 检索 + 惰性 VLM）...", flush=True)
agent = ToolCallAgent()


def chat_fn(message, history):
    """多轮对话：复用 agent 持久历史。"""
    ans = agent.chat(message)
    history = history or []
    history.append((message, ans))
    return "", history


def image_fn(img_a, img_b):
    if img_b is not None and img_a is not None:
        return agent._exec("detect_change", {"image_a": img_a, "image_b": img_b})
    if img_a is not None:
        return agent._exec("analyze_image", {"image_path": img_a})
    return "请上传影像"


def report_fn(method, dataset, metrics, change_map):
    if not change_map:
        return "请上传变化图"
    return agent._exec("generate_report", {"method": method, "dataset": dataset,
                                           "metrics": metrics, "change_map": change_map})


with gr.Blocks(title="领域文献阅读 Agent") as demo:
    gr.Markdown("# 📚 领域文献阅读 Agent\n面向研究生新生的文献速读助手（遥感变化检测示例）")

    with gr.Tab("💬 论文问答"):
        gr.Markdown("问任何领域问题，agent 会检索论文并给出带引用的回答（支持追问）。")
        chatbot = gr.Chatbot(label="对话")
        msg = gr.Textbox(label="问题", placeholder="例如：变化检测常用哪些评价指标？")
        clear = gr.Button("清空对话")
        msg.submit(chat_fn, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: (None, None), None, [msg, chatbot])

    with gr.Tab("🛰️ 影像分析"):
        gr.Markdown("上传 1 张影像=描述；上传 2 张=双时相变化检测。")
        img_a = gr.Image(label="影像 A（T1）", type="filepath")
        img_b = gr.Image(label="影像 B（T2，可选）", type="filepath")
        img_btn = gr.Button("分析")
        img_out = gr.Textbox(label="结果", lines=10)
        img_btn.click(image_fn, [img_a, img_b], [img_out])

    with gr.Tab("📝 实验报告"):
        gr.Markdown("输入方法/数据集/指标 + 上传变化图，自动生成结果分析报告。")
        method = gr.Textbox(label="方法", value="ChangeFormer")
        dataset = gr.Textbox(label="数据集", value="LEVIR-CD")
        metrics = gr.Textbox(label="指标", value="F1=0.892, IoU=0.805, OA=0.990, Precision=0.910, Recall=0.874")
        change_map = gr.Image(label="变化图（mask）", type="filepath")
        rpt_btn = gr.Button("生成报告")
        rpt_out = gr.Textbox(label="报告", lines=15)
        rpt_btn.click(report_fn, [method, dataset, metrics, change_map], [rpt_out])

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
