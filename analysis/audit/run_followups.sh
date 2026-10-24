#!/usr/bin/env bash
# Follow-up runs for the revised paper, four models one after another on one GPU: off-target necessity, steering
# with directions fit on the text alone and at the answer (nine concepts), and necessity of the register-matched
# direction (Llama and Qwen only).
#   GPU=0 setsid nohup bash run_followups.sh > followups.log 2>&1 &
# BLAS threads are capped: on a container with a CPU quota, uncapped OpenBLAS made the probe and null stages of
# run_audit.py about 50 times slower (same results).
set -u
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0} MI_PROJECT_ROOT="$ROOT" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
OUT="$ROOT/results"
C9=suicide,suicide_depression,toxicity,sentiment,sports,business,refusal,truth,anger
for M in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3 \
         unsloth/gemma-2-9b-it; do
  echo "=== $(date -u +%FT%TZ) $M"
  DT=float16; B=16; case "$M" in *gemma*) DT=bfloat16; B=8;; esac
  python offtarget_necessity.py --model "$M" --dtype "$DT" --batch "$B" \
    --out-dir "$OUT/offtarget_necessity" || echo "!!! $M offtarget_necessity failed"
  python steering_mech.py --model "$M" --dtype "$DT" --batch "$B" --followup --fit-prompt post --concepts "$C9" \
    --kappas 0.1 --only-variants orig,minus_yes --out-dir "$OUT/steering_post" || echo "!!! $M steering_post failed"
  python steering_mech.py --model "$M" --dtype "$DT" --batch "$B" --followup --fit-prompt question --concepts "$C9" \
    --kappas 0.1 --only-variants orig,minus_yes,interchange --out-dir "$OUT/steering_q9" || echo "!!! $M steering_q9 failed"
  case "$M" in
    *Llama*|*Qwen*)
      python run_audit.py --concept suicide_depression --model "$M" --dtype "$DT" --batch "$B" --n-fit 2000 \
        --n-eval 1000 --n-nec 300 --n-null 10 --force-ladder --skip-unmatched-nulls --skip-sufficient \
        --out-dir "$OUT/audit_swmh" || echo "!!! $M run_audit failed" ;;
  esac
done
echo "=== $(date -u +%FT%TZ) done"
