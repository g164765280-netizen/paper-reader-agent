import sys
sys.path.insert(0, ".")
from core.kg_store import KGStore

kg = KGStore()
kg.load_from_json()
print("图谱节点:", len(kg.graph.nodes), "边:", len(kg.graph.edges), flush=True)
r = kg.generate_community_reports()
print("社区摘要结果:", r, flush=True)