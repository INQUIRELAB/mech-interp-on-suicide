#!/usr/bin/env bash
# yes_bias_audit.py on the four paper models, one after another on one GPU.
#   GPUS=0 setsid nohup bash run_yes_bias.sh > yes_bias.log 2>&1 &
# SMOKE=1 runs a fast smoke test on Llama only (checks loaders, prompts and output paths).
set -eu
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${GPUS:-0} MI_PROJECT_ROOT="$ROOT" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
if [ "${SMOKE:-0}" = 1 ]; then
  "$PY" yes_bias_audit.py --model unsloth/Meta-Llama-3.1-8B-Instruct --smoke
  exit
fi
for M in unsloth/Meta-Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct \
         mistralai/Mistral-7B-Instruct-v0.3 unsloth/gemma-2-9b-it; do
  echo "=== $(date -u +%FT%TZ) $M"
  DT=float16; case "$M" in *gemma*) DT=bfloat16;; esac   # Gemma-2 overflows in fp16 (as in run_prompts.sh)
  "$PY" yes_bias_audit.py --model "$M" --dtype "$DT" || echo "!!! $M failed"
done
echo "=== $(date -u +%FT%TZ) done"
