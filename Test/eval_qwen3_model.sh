#!/usr/bin/env bash

# Usage: ./eval_qwen3_model.sh [EXTRA_ARGS]
# Example:
#   ./eval_qwen3_model.sh --sample_num 200 --num_beams 30

# Qwen3模型路径配置
BASE_MODEL_DEFAULT="../Qwen3/model/Qwen3-1-7B-expanded-vocab"
LORA_MODEL_DEFAULT="../Qwen3/results/sid_mapping_model"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-$BASE_MODEL_DEFAULT}"
LORA_MODEL_PATH="${LORA_MODEL_PATH:-$LORA_MODEL_DEFAULT}"

# 创建日志目录
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/qwen3_eval_${TS}.log"

echo "🚀 Starting Qwen3 evaluation..."
echo "📝 Log file: $LOG"
echo "⏰ Started at: $(date)"

# 设置环境变量
export TOKENIZERS_PARALLELISM=false

# 使用nohup后台运行，所有输出都到一个日志文件
nohup python3 -u test_qwen3_hitrate.py \
  --base_model_path "${BASE_MODEL_PATH}" \
  --lora_model_path "${LORA_MODEL_PATH}" \
  --data_path . \
  --dataset Beauty \
  --test_batch_size 8 \
  --num_beams 20 \
  --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10" \
  --think_max_tokens 0 \
  --print_generations \
  --sample_num 400 \
  --log_file "$LOG" \
  --filter_items \
  "$@" > "$LOG" 2>&1 &

# 获取进程ID
PID=$!
echo "🔄 Process ID: $PID"
echo "📋 All output in: $LOG"

echo ""
echo "Commands to monitor progress:"
echo "  tail -f $LOG        # 查看日志"
echo "  kill $PID           # 终止进程"
echo "  ps aux | grep $PID  # 检查进程状态"