#!/usr/bin/env bash

# Stage 2 八GPU加速测试脚本
# 针对8GPU机器优化配置

# Stage 2模型路径配置
BASE_MODEL_DEFAULT="../Qwen3/model/Qwen3-1-7B-expanded-vocab"
STAGE1_MODEL_DEFAULT="../Qwen3/results/sid_mapping_model"
STAGE2_MODEL_DEFAULT="../Qwen3/results/stage2_recommendation_model"
STAGE2_VAL_DATA_DEFAULT="../Qwen3/data_stage2/val.parquet"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-$BASE_MODEL_DEFAULT}"
STAGE1_MODEL_PATH="${STAGE1_MODEL_PATH:-$STAGE1_MODEL_DEFAULT}"
STAGE2_MODEL_PATH="${STAGE2_MODEL_PATH:-$STAGE2_MODEL_DEFAULT}"
STAGE2_VAL_DATA_PATH="${STAGE2_VAL_DATA_PATH:-$STAGE2_VAL_DATA_DEFAULT}"

# 创建日志目录
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/stage2_8gpu_${TS}.log"

echo "🚀 Starting Qwen3 Stage 2 evaluation with 8 GPUs..."
echo "📝 Log file: $LOG"
echo "⏰ Started at: $(date)"

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