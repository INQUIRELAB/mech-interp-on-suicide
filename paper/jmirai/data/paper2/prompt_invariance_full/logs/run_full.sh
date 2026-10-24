#!/usr/bin/env bash
# Full-text rerun of matched_prompts.py on SWMH (r/SuicideWatch vs r/depression): body budget 4096 tokens (no post cut).
set -u
cd /root/rev_full/analysis/audit
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-4} HF_HUB_OFFLINE=1 MI_PROJECT_ROOT=/root/rev_full PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
OUT=/root/rev_full/results/prompt_invariance_full
mkdir -p $OUT /root/rev_full/logs
run() {  # model dtype max_batch_tokens log
  python matched_prompts.py --model "$1" --dtype "$2" --body-tokens 4096 --max-batch-tokens "$3" --batch 16 \
    --out-suffix _full --out-dir $OUT > /root/rev_full/logs/full_$4.log 2>&1
  echo "$4 exit $? $(date -u +%FT%TZ)" >> /root/rev_full/logs/status
}
run meta-llama/Llama-3.1-8B-Instruct float16 16384 llama
for i in $(seq 1 360); do [ -f /root/models/gemma-2-9b-it/.done ] && break; sleep 30; done
run /root/models/Qwen2.5-7B-Instruct float16 16384 qwen
run /root/models/Mistral-7B-Instruct-v0.3 float16 16384 mistral
run /root/models/gemma-2-9b-it bfloat16 8192 gemma
echo "all done $(date -u +%FT%TZ)" >> /root/rev_full/logs/status
