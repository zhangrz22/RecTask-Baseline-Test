#!/usr/bin/env bash

# Usage: ./Test/eval_model.sh [EXTRA_ARGS]
# Example:
#   ./Test/eval_model.sh --test_batch_size 2 --num_beams 20

CKPT_DEFAULT="/llm-reco-ssd-share/baohonghui/think_pretrain/results/pretrain_only_te/hf_model_step3234_final"
CKPT_PATH="$CKPT_DEFAULT"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/eval_${TS}.log

export TOKENIZERS_PARALLELISM=false
nohup python3 -u test_hitrate.py \
  --ckpt_path "${CKPT_PATH}" \
  --filter_items \
  --data_path . \
  --dataset Beauty \
  --test_batch_size 1 \
  --num_beams 16 \
  --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10" \
  --enable_cot \
  --think_max_tokens 64 \
  --print_generations \
  --sample_num 100 \
  "$@" > "$LOG" 2>&1 &

echo "Started eval. Logs: $LOG"
