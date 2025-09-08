#!/bin/bash

# Prepare SID-to-Title mapping dataset

echo "Preparing SID-to-Title mapping dataset..."

python3 prepare_dataset.py \
    --index_file ../Test/Beauty/Beauty.index.json \
    --item_file ../Test/Beauty/Beauty.item.json \
    --output_dir ./data \
    --format parquet \
    --seed 42

echo "Dataset preparation completed!"
echo "Files created:"
echo "- ./data/train.parquet"
echo "- ./data/val.parquet" 
echo "- ./data/test.parquet"