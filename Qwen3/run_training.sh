#!/bin/bash

# SID-to-Title mapping fine-tuning with DeepSpeed

echo "Starting SID-to-Title mapping fine-tuning..."

# Create necessary directories
mkdir -p ./results/sid_mapping_model
mkdir -p ./logs/sid_mapping

# DeepSpeed config (create if not exists)
if [ ! -f "./ds_config_zero2.json" ]; then
    cat > ./ds_config_zero2.json << 'EOF'
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

# Run training with DeepSpeed
nohup deepspeed --hostfile=./hostfile \
    --num_gpus 8 train_sid_mapping.py \
    --model_name_or_path ./model/Qwen3-1-7B-expanded-vocab \
    --data_dir ./data \
    --train_file train.parquet \
    --validation_file val.parquet \
    --max_seq_length 256 \
    --max_token_range 256 \
    --output_dir ./results/sid_mapping_model \
    --logging_dir ./logs/sid_mapping \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
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
    --deepspeed ./ds_config_zero2.json \
    --report_to none \
    >> sid_mapping_training.log 2>&1 &

echo "Training started in background!"
echo "Check progress: tail -f sid_mapping_training.log"
echo "Model will be saved to: ./results/sid_mapping_model"