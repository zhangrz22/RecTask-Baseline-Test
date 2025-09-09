#!/usr/bin/env bash

# Stage 2 八GPU加速测试脚本
# 针对8GPU机器优化配置

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
LOG="logs/stage2_8gpu_${TS}.log"

echo "🚀 Starting Qwen3 Stage 2 evaluation with 8 GPUs..."
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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 显示GPU状态
echo ""
echo "📊 GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits

echo ""
echo "🔧 Configuration:"
echo "  Batch size per GPU: 1 (total: 8)"
echo "  Sample size: 400"
echo "  Beam search: 20"
echo ""

# 使用torchrun启动8GPU测试
torchrun \
  --nproc_per_node=8 \
  --master_port=29502 \
  test_stage2_hitrate.py \
  --base_model_path "${BASE_MODEL_PATH}" \
  --stage1_model_path "${STAGE1_MODEL_PATH}" \
  --stage2_model_path "${STAGE2_MODEL_PATH}" \
  --stage2_val_data_path "${STAGE2_VAL_DATA_PATH}" \
  $PREPROCESS_ARGS \
  --test_batch_size 1 \
  --num_beams 20 \
  --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10" \
  --enable_cot \
  --think_max_tokens 64 \
  --print_generations \
  --sample_num 1007 \
  --log_file "$LOG" \
  --filter_items \
  "$@" 2>&1 | tee "$LOG"

echo ""
echo "⏰ Finished at: $(date)"
echo ""
echo "📋 Performance Summary:"
echo "  Log file: $LOG"
echo "  Final results: tail -30 $LOG"
echo ""
echo "✅ 8-GPU Stage 2 testing completed!"