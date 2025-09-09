#!/usr/bin/env bash

# Usage: ./run_stage2_test.sh [EXTRA_ARGS]
# Example:
#   ./run_stage2_test.sh --sample_num 200 --num_beams 30

# Stage 2模型路径配置
BASE_MODEL_DEFAULT="../Qwen3/model/Qwen3-1-7B-expanded-vocab"
STAGE1_MODEL_DEFAULT="../Qwen3/results/sid_mapping_model"
STAGE2_MODEL_DEFAULT="../Qwen3/results/stage2_recommendation_model"
STAGE2_VAL_DATA_DEFAULT="../Qwen3/data_stage2/val.parquet"
PREPROCESSED_DATA_DEFAULT="../Qwen3/data_stage2/val_preprocessed.json"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-$BASE_MODEL_DEFAULT}"
STAGE1_MODEL_PATH="${STAGE1_MODEL_PATH:-$STAGE1_MODEL_DEFAULT}"
STAGE2_MODEL_PATH="${STAGE2_MODEL_PATH:-$STAGE2_MODEL_DEFAULT}"
STAGE2_VAL_DATA_PATH="${STAGE2_VAL_DATA_PATH:-$STAGE2_VAL_DATA_DEFAULT}"
PREPROCESSED_DATA_PATH="${PREPROCESSED_DATA_PATH:-$PREPROCESSED_DATA_DEFAULT}"

# 是否使用预处理数据（默认开启以提高性能）
USE_PREPROCESSED="${USE_PREPROCESSED:-true}"

# 创建日志目录
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/stage2_eval_${TS}.log"

echo "🚀 Starting Qwen3 Stage 2 evaluation..."
echo "📝 Log file: $LOG"
echo "⏰ Started at: $(date)"

# 检查预处理数据是否存在
if [ "$USE_PREPROCESSED" = "true" ]; then
    if [ -f "$PREPROCESSED_DATA_PATH" ]; then
        echo "✅ Using preprocessed data: $PREPROCESSED_DATA_PATH"
        PREPROCESS_ARGS="--use_preprocessed --preprocessed_data_path $PREPROCESSED_DATA_PATH"
    else
        echo "⚠️  Preprocessed data not found at: $PREPROCESSED_DATA_PATH"
        echo "🔄 Creating preprocessed data first..."
        python3 preprocess_stage2_data.py \
            --input_path "$STAGE2_VAL_DATA_PATH" \
            --output_path "$PREPROCESSED_DATA_PATH"
        if [ $? -eq 0 ]; then
            echo "✅ Preprocessed data created successfully"
            PREPROCESS_ARGS="--use_preprocessed --preprocessed_data_path $PREPROCESSED_DATA_PATH"
        else
            echo "❌ Failed to create preprocessed data, using raw data"
            PREPROCESS_ARGS=""
        fi
    fi
else
    echo "⚠️  Using raw data (slower loading)"
    PREPROCESS_ARGS=""
fi

# 设置环境变量
export TOKENIZERS_PARALLELISM=false

# 使用nohup后台运行，所有输出都到一个日志文件
nohup python3 -u test_stage2_hitrate.py \
  --base_model_path "${BASE_MODEL_PATH}" \
  --stage1_model_path "${STAGE1_MODEL_PATH}" \
  --stage2_model_path "${STAGE2_MODEL_PATH}" \
  --stage2_val_data_path "${STAGE2_VAL_DATA_PATH}" \
  $PREPROCESS_ARGS \
  --test_batch_size 1 \
  --num_beams 20 \
  --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10" \
  --think_max_tokens 0 \
  --print_generations \
  --sample_num 100 \
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