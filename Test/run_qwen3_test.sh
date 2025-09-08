#!/bin/bash

# Qwen3 Hit Rate测试运行脚本

echo "🧪 Starting Qwen3 Hit Rate Test..."

# 创建结果目录
mkdir -p ./results

# 运行测试
python3 test_qwen3_hitrate.py \
    --data_path ./ \
    --dataset Beauty \
    --base_model_path ../Qwen3/model/Qwen3-1-7B-expanded-vocab \
    --lora_model_path ../Qwen3/results/sid_mapping_model \
    --test_batch_size 1 \
    --num_beams 20 \
    --sample_num 100 \
    --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10" \
    --results_file ./results/qwen3_hitrate_results.json \
    --log_file ./results/qwen3_test_detailed.log \
    --enable_cot \
    --think_max_tokens 64 \
    --print_generations

echo "✅ Test completed!"
echo "📊 Results: ./results/qwen3_hitrate_results.json"
echo "📝 Detailed log: ./results/qwen3_test_detailed.log"