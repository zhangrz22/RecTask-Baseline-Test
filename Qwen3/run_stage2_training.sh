#!/bin/bash

# 第二阶段推荐训练脚本

echo "🚀 Starting Stage 2 Recommendation Training..."

# 创建必要目录
mkdir -p ./results/stage2_recommendation_model
mkdir -p ./logs/stage2

# DeepSpeed配置文件
if [ ! -f "./ds_config_stage2.json" ]; then
    cat > ./ds_config_stage2.json << 'EOF'
{
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto", 
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
EOF
fi

# 记录开始时间
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/stage2/training_${TS}.log"

echo "⏰ Started at: $(date)"
echo "📝 Log file: $LOG"

# 运行训练
nohup deepspeed --hostfile=./hostfile \
    --num_gpus 8 train_recommendation_stage2.py \
    --base_model_path ./model/Qwen3-1-7B-expanded-vocab \
    --stage1_lora_path ./results/sid_mapping_model \
    --max_token_range 256 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --data_dir ./data_stage2 \
    --max_seq_length 1024 \
    --train_file train.parquet \
    --validation_file val.parquet \
    --output_dir ./results/stage2_recommendation_model \
    --logging_dir ./logs/stage2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --bf16 true \
    --gradient_checkpointing true \
    --deepspeed ./ds_config_stage2.json \
    --report_to none \
    >> "$LOG" 2>&1 &

PID=$!
echo "🔄 Process ID: $PID"
echo "📋 Training log: $LOG"

echo ""
echo "Commands to monitor progress:"
echo "  tail -f $LOG        # 查看训练日志"
echo "  kill $PID           # 终止训练"
echo "  ps aux | grep $PID  # 检查进程状态"

echo ""
echo "Training started in background!"
echo "Model will be saved to: ./results/stage2_recommendation_model"