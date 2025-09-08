#!/bin/bash

# SaSRec Training Script
# This script starts training with optimized parameters for the dataset

# Create results directory if it doesn't exist
mkdir -p results

# Training parameters
DATASET="data_long_format"
TRAIN_DIR="./results"
DEVICE="cuda"  # Change to "cpu" if no GPU available
BATCH_SIZE=128
LR=0.001
MAX_LEN=200
HIDDEN_UNITS=64
NUM_BLOCKS=4
NUM_HEADS=1
NUM_EPOCHS=200
DROPOUT_RATE=0.1
L2_EMB=0.0

echo "Starting SaSRec training..."
echo "Dataset: $DATASET"
echo "Results will be saved to: $TRAIN_DIR"
echo "Device: $DEVICE"
echo ""

# Run training
python main.py \
    --dataset=$DATASET \
    --train_dir=$TRAIN_DIR \
    --batch_size=$BATCH_SIZE \
    --lr=$LR \
    --maxlen=$MAX_LEN \
    --hidden_units=$HIDDEN_UNITS \
    --num_blocks=$NUM_BLOCKS \
    --num_heads=$NUM_HEADS \
    --num_epochs=$NUM_EPOCHS \
    --dropout_rate=$DROPOUT_RATE \
    --l2_emb=$L2_EMB \
    --device=$DEVICE

echo ""
echo "Training completed!"
echo "Check results in: $TRAIN_DIR"
echo "- Log file: $TRAIN_DIR/log.txt"
echo "- Model checkpoints: $TRAIN_DIR/*.pth"
echo "- Training arguments: $TRAIN_DIR/args.txt"