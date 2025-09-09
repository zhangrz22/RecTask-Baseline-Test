#!/bin/bash

# 第二阶段推荐数据准备脚本

echo "🔄 Starting Stage 2 data preparation..."

# 创建必要目录
mkdir -p ./data_stage2
mkdir -p ./logs/stage2

# 记录开始时间
echo "⏰ Started at: $(date)"

# 运行数据准备
python3 prepare_recommendation_data.py \
    --data_path ../Test/Beauty \
    --inter_file Beauty.inter.json \
    --index_file Beauty.index.json \
    --output_dir ./data_stage2 \
    --min_seq_len 3 \
    --his_sep ", " \
    --seed 42

if [ $? -eq 0 ]; then
    echo "✅ Data preparation completed!"
    echo "📊 Check data statistics:"
    ls -la ./data_stage2/
    
    echo ""
    echo "📝 Data config:"
    cat ./data_stage2/data_config.json
else
    echo "❌ Data preparation failed!"
    exit 1
fi

echo "⏰ Finished at: $(date)"