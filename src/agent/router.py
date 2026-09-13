# -*- coding: utf-8 -*-
"""路由器：根据各插件 can_handle 评分，把问题分流到最合适的插件。

「一切皆插件」：路由器只认 Plugin 契约，不关心插件内部实现。
新增插件只需实现 can_handle + run，注册进插件列表即可。
"""
from __future__ import annotations

from plugin_base import PluginResult


class Router:
    def __init__(self, plugins: list):
        self.plugins = plugins

    def route(self, question: str) -> tuple[str, float]:
        """返回 (插件名, 适合度) 。取 can_handle 最高分的插件。"""
        best_name, best_score = None, -1.0
        for p in self.plugins:
            s = p.can_handle(question)
            if s > best_score:
                best_name, best_score = p.name, s
        return best_name, best_score

    def answer(self, question: str, **kwargs) -> PluginResult:
        name, score = self.route(question)
        for p in self.plugins:
            if p.name == name:
                r = p.run(question, **kwargs)
                r.meta["routed_tool"] = name
                r.meta["route_score"] = round(score, 3)
                return r
        return PluginResult(answer="没有匹配的插件", refused=True,
                            meta={"routed_tool": None})