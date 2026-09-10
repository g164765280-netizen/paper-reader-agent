#!/bin/bash
# 等待 ingest 完成，然后自动跑社区摘要 + 专家评测
set -e
cd /media/sdb1/gzj/project/llm/reference/litkg-assistant-main
LOG=ingest_full.log
PY=/home/mcislab/anaconda3/envs/all-in-rag/bin/python

echo "[$(date +%H:%M:%S)] 等待 ingest 完成..."
while ! grep -q '^DONE' $LOG 2>/dev/null; do
  n=$(grep -c 'ok (' $LOG 2>/dev/null || echo 0)
  echo "  ... 已完成 $n/109 篇 $(date +%H:%M:%S)"
  sleep 300
done
echo "[$(date +%H:%M:%S)] ingest 完成，生成社区摘要..."

$PY -u gen_community.py 2>&1 | tee community.log

echo "[$(date +%H:%M:%S)] 社区摘要完成，运行专家评测..."
$PY -u /media/sdb1/gzj/support/llm/rscd/eval_expert.py 2>&1 | tee expert_eval.log

echo "[$(date +%H:%M:%S)] 全部完成"