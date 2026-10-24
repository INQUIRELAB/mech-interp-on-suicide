#!/usr/bin/env bash
# matched_prompts.py on full posts (body budget 4096 tokens; no register-matched post is cut), four models on one GPU.
#   GPU=0 setsid nohup bash run_full_text.sh > full_text.log 2>&1 &
set -u
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0} MI_PROJECT_ROOT="$ROOT" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
OUT="$ROOT/results/paper2/prompt_invariance_full"
for spec in meta-llama/Llama-3.1-8B-Instruct:float16:16384 Qwen/Qwen2.5-7B-Instruct:float16:16384 \
            mistralai/Mistral-7B-Instruct-v0.3:float16:16384 google/gemma-2-9b-it:bfloat16:8192; do
  IFS=: read -r M DT TOK <<< "$spec"
  echo "=== $(date -u +%FT%TZ) $M"
  python matched_prompts.py --model "$M" --dtype "$DT" --body-tokens 4096 --max-batch-tokens "$TOK" --batch 16 \
    --out-suffix _full --out-dir "$OUT" || echo "!!! $M failed"
done
