#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/media/sdb1/gzj
SUPPORT="$ROOT/support/llm/rscd"
ENV="$ROOT/support/llm/envs/medicalgpt"
PY="$ENV/bin/python"
BASE="$ROOT/weights/llm/rscd/qwen25_7b_rscd_sft_merge"
TRAIN_SCRIPT="$ROOT/support/llm/medicalgpt/full_20260814/dpo_training_7b_offline_attempt2.py"
SOURCE="$ROOT/data/rscd/reward/rscd_dpo.jsonl"
WEIGHTS="$ROOT/weights/llm/rscd/qwen25_7b_rscd_dpo"
RESULT="$ROOT/generated_data/llm/rscd/preference/rscd_qwen25_7b_dpo"

for p in "$PY" "$BASE" "$SOURCE" "$TRAIN_SCRIPT"; do
  test -e "$p" || { echo "missing: $p" >&2; exit 2; }
done
test ! -e "$WEIGHTS" || { echo "refusing overwrite: $WEIGHTS" >&2; exit 2; }
test ! -e "$RESULT" || { echo "refusing overwrite: $RESULT" >&2; exit 2; }
gpu_rows=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')
test -z "$gpu_rows" || { echo "GPU busy" >&2; exit 2; }

mkdir -p "$RESULT"/{data/train,data/eval,metrics,logs} "$WEIGHTS" "$SUPPORT/cache_dpo"

"$PY" - "$SOURCE" "$RESULT" <<'PY'
import json, pathlib, sys
source = pathlib.Path(sys.argv[1]); root = pathlib.Path(sys.argv[2])
lines = source.read_text(encoding="utf-8").splitlines()
order = [(i * 137) % len(lines) for i in range(len(lines))]
train_ids, eval_ids = order[:256], order[256:320]
assert not (set(train_ids) & set(eval_ids))
for split, ids in (("train", train_ids), ("eval", eval_ids)):
    (root/"data"/split/"rscd_dpo.jsonl").write_text("".join(lines[i]+"\n" for i in ids), encoding="utf-8")
(root/"MANIFEST.txt").write_text(f"source: {source}\nmethod: Qwen2.5-7B RS-CD DPO on merged SFT, 50 steps, BF16 LoRA r=8, single GPU\ndate: 20260906\nsplit: 256 train / 64 eval\n", encoding="utf-8")
(root/"STATUS.json").write_text(json.dumps({"status":"running","stage":"train"}, indent=2)+"\n")
PY

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/project/llm/MedicalGPT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMAND=(
 "$PY" "$TRAIN_SCRIPT"
 --model_name_or_path "$BASE" --tokenizer_name_or_path "$BASE"
 --train_file_dir "$RESULT/data/train" --validation_file_dir "$RESULT/data/eval"
 --output_dir "$WEIGHTS" --cache_dir "$SUPPORT/cache_dpo"
 --do_train --use_peft True
 --max_train_samples 256 --max_eval_samples 64
 --max_source_length 512 --max_target_length 256
 --per_device_train_batch_size 1 --per_device_eval_batch_size 1
 --max_steps 50 --learning_rate 5e-6 --warmup_steps 5 --weight_decay 0.05
 --gradient_accumulation_steps 4
 --logging_steps 1 --eval_strategy no --save_steps 50
 --preprocessing_num_workers 1 --overwrite_cache True
 --target_modules all --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05
 --torch_dtype bfloat16 --bf16 True --fp16 False --gradient_checkpointing True
 --report_to none --remove_unused_columns False --no_trust_remote_code --tool_format default
)

set +e
"${COMMAND[@]}" 2>&1 | tee "$RESULT/logs/train.log"
ec=${PIPESTATUS[0]}
set -e
echo "$ec" > "$RESULT/metrics/train_exit_code.txt"
if test "$ec" -ne 0; then echo "DPO FAILED ec=$ec"; exit "$ec"; fi
test -s "$WEIGHTS/adapter_model.safetensors"
"$PY" - "$RESULT" "$WEIGHTS" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); w=pathlib.Path(sys.argv[2])
tr=json.loads((w/"train_results.json").read_text())
st=json.loads((w/"trainer_state.json").read_text())
summary={"stage":"Qwen2.5-7B RS-CD DPO LoRA","status":"completed","global_step":st["global_step"],
 "train_loss":tr["train_loss"],"train_runtime":tr["train_runtime"]}
(root/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
(root/"STATUS.json").write_text(json.dumps({"status":"completed","stage":"complete","global_step":st["global_step"],"exit_code":0},indent=2)+"\n")
print(json.dumps(summary,ensure_ascii=False))
PY
echo "DPO_DONE weights=$WEIGHTS result=$RESULT"
