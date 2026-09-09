#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/media/sdb1/gzj
SUPPORT="$ROOT/support/llm/rscd"
ENV="$ROOT/support/llm/envs/medicalgpt"
PY="$ENV/bin/python"
TORCHRUN="$ENV/bin/torchrun"
BASE="/home/mcislab/weights/Qwen2.5-7B-Instruct"
TRAIN_SCRIPT="$ROOT/support/llm/medicalgpt/sft_7b_20260814/supervised_finetuning_7b_offline.py"
SOURCE="$ROOT/data/rscd/sft/rscd_sft_v2.jsonl"
WEIGHTS="$ROOT/weights/llm/rscd/qwen25_7b_rscd_sft_lora_v2"
RESULT="$ROOT/generated_data/llm/rscd/sft/rscd_qwen25_7b_sft_lora_v2"

for p in "$PY" "$TORCHRUN" "$BASE" "$SOURCE" "$TRAIN_SCRIPT"; do
  test -e "$p" || { echo "missing: $p" >&2; exit 2; }
done
test ! -e "$WEIGHTS" || { echo "refusing overwrite: $WEIGHTS" >&2; exit 2; }
test ! -e "$RESULT" || { echo "refusing overwrite: $RESULT" >&2; exit 2; }
gpu_rows=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')
test -z "$gpu_rows" || { echo "GPU busy" >&2; exit 2; }

mkdir -p "$RESULT"/{data/train,data/eval,metrics,logs} "$WEIGHTS" "$SUPPORT/cache"

"$PY" - "$SOURCE" "$RESULT" <<'PY'
import json, pathlib, sys
source = pathlib.Path(sys.argv[1]); root = pathlib.Path(sys.argv[2])
lines = source.read_text(encoding="utf-8").splitlines()
order = [(i * 137) % len(lines) for i in range(len(lines))]
train_ids, eval_ids = order[:512], order[512:576]
assert not (set(train_ids) & set(eval_ids))
for split, ids in (("train", train_ids), ("eval", eval_ids)):
    (root/"data"/split/"rscd_sft_v2.jsonl").write_text("".join(lines[i]+"\n" for i in ids), encoding="utf-8")
(root/"MANIFEST.txt").write_text(f"source: {source}\nmethod: Qwen2.5-7B RS-CD SFT v2 BF16 LoRA r=8, 200 steps, dual 4090 DDP\nsplit: 512 train / 64 eval\n", encoding="utf-8")
(root/"STATUS.json").write_text(json.dumps({"status":"running","stage":"train"}, indent=2)+"\n")
PY

export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/project/llm/MedicalGPT"

COMMAND=(
  "$TORCHRUN" --nproc_per_node 2 --master_port 29516 "$TRAIN_SCRIPT"
  --model_name_or_path "$BASE" --tokenizer_name_or_path "$BASE"
  --train_file_dir "$RESULT/data/train" --validation_file_dir "$RESULT/data/eval"
  --output_dir "$WEIGHTS" --cache_dir "$SUPPORT/cache"
  --do_train --do_eval --use_peft True
  --max_train_samples 512 --max_eval_samples 64
  --model_max_length 512 --per_device_train_batch_size 1 --per_device_eval_batch_size 1
  --max_steps 200 --learning_rate 2e-5 --warmup_steps 20 --weight_decay 0.05
  --gradient_accumulation_steps 4 --logging_strategy steps --logging_steps 10 --logging_first_step True
  --eval_strategy steps --eval_steps 100 --save_strategy steps --save_steps 200 --save_total_limit 1
  --preprocessing_num_workers 1 --overwrite_cache True
  --target_modules all --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05
  --torch_dtype bfloat16 --bf16 --gradient_checkpointing True
  --ddp_find_unused_parameters False --ddp_timeout 30000
  --seed 42 --report_to none --no_trust_remote_code --tool_format default
)

set +e
"${COMMAND[@]}" 2>&1 | tee "$RESULT/logs/train.log"
ec=${PIPESTATUS[0]}
set -e
echo "$ec" > "$RESULT/metrics/train_exit_code.txt"
if test "$ec" -ne 0; then echo "SFT_V2 FAILED ec=$ec"; exit "$ec"; fi
test -s "$WEIGHTS/adapter_model.safetensors"
"$PY" - "$RESULT" "$WEIGHTS" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); w=pathlib.Path(sys.argv[2])
tr=json.loads((w/"train_results.json").read_text()); ev=json.loads((w/"eval_results.json").read_text())
st=json.loads((w/"checkpoint-200/trainer_state.json").read_text())
summary={"stage":"Qwen2.5-7B RS-CD SFT v2 LoRA","status":"completed","global_step":st["global_step"],
 "train_loss":tr["train_loss"],"eval_loss":ev["eval_loss"],"train_runtime":tr["train_runtime"]}
(root/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
(root/"STATUS.json").write_text(json.dumps({"status":"completed","stage":"complete","global_step":st["global_step"],"exit_code":0},indent=2)+"\n")
print(json.dumps(summary,ensure_ascii=False))
PY
echo "SFT_V2_DONE weights=$WEIGHTS result=$RESULT"
