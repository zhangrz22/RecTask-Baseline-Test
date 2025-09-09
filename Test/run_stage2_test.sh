#!/bin/bash

# Stage 2 推荐模型测试脚本

echo "🧪 Starting Qwen3 Stage 2 Hit Rate Test..."

# 创建日志目录
mkdir -p ./logs

# 记录开始时间
echo "⏰ Started at: $(date)"

echo ""
echo "🎯 Testing Stage 2 model only..."
python3 test_stage2_hitrate.py \
    --data_path ./Beauty \
    --dataset Beauty \
    --index_file Beauty.index.json \
    --base_model_path ../Qwen3/model/Qwen3-1-7B-expanded-vocab \
    --stage1_model_path ../Qwen3/results/sid_mapping_model \
    --stage2_model_path ../Qwen3/results/stage2_recommendation_model \
    --test_batch_size 8 \
    --num_beams 20 \
    --sample_num 1000 \
    --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10" \
    --enable_cot \
    --think_max_tokens 64 \
    --print_generations \
    --log_file ./logs/stage2_test_only.log

echo ""
echo "📊 Testing Stage 2 vs Stage 1 comparison..."
python3 test_stage2_hitrate.py \
    --data_path ./Beauty \
    --dataset Beauty \
    --index_file Beauty.index.json \
    --stage2_model_path ../Qwen3/results/stage2_recommendation_model \
    --compare_with_stage1 \
    --stage1_model_path ../Qwen3/results/sid_mapping_model \
    --base_model_path ../Qwen3/model/Qwen3-1-7B-expanded-vocab \
    --test_batch_size 8 \
    --num_beams 20 \
    --sample_num 1000 \
    --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10" \
    --enable_cot \
    --think_max_tokens 64 \
    --log_file ./logs/stage2_vs_stage1_comparison.log

echo ""
echo "⏰ Finished at: $(date)"
echo ""
echo "📋 Check results:"
echo "  Stage 2 only: tail -50 ./logs/stage2_test_only.log"
echo "  Comparison: tail -50 ./logs/stage2_vs_stage1_comparison.log"
echo ""
echo "✅ Stage 2 testing completed!"