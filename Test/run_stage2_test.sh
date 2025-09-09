#!/bin/bash

# Stage 2 推荐模型测试脚本
# 测试完整的 Base + Stage1 + Stage2 模型效果

echo "🧪 Testing Qwen3 Stage 2 Model with Validation Data..."
echo "    使用训练时预留的验证数据进行测试"

# 创建日志目录
mkdir -p ./logs

# 记录开始时间
echo "⏰ Started at: $(date)"

echo ""
echo "📊 Testing Complete Stage 2 Model (Base + Stage1 + Stage2)..."
echo "    数据源: 训练时预留的验证数据"
echo "    架构: Base Model + Stage1 LoRA (merged) + Stage2 LoRA"

python3 test_stage2_hitrate.py \
    --base_model_path ../Qwen3/model/Qwen3-1-7B-expanded-vocab \
    --stage1_model_path ../Qwen3/results/sid_mapping_model \
    --stage2_model_path ../Qwen3/results/stage2_recommendation_model \
    --stage2_val_data_path ../Qwen3/data_stage2/val.parquet \
    --test_batch_size 8 \
    --num_beams 20 \
    --sample_num 1000 \
    --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10" \
    --enable_cot \
    --think_max_tokens 64 \
    --print_generations \
    --log_file ./logs/stage2_test.log

echo ""
echo "⏰ Finished at: $(date)"
echo ""
echo "📋 Check results:"
echo "  tail -50 ./logs/stage2_test.log"
echo ""
echo "✅ Stage 2 testing completed!"