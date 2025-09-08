#!/bin/bash

# Qwen3 Vocabulary Expansion Script
# This script expands Qwen3-1.7B with recommendation-specific tokens

echo "Starting Qwen3 vocabulary expansion..."

# Configuration
BASE_MODEL="/home/liuzhanyu/SidTextAlign/content-rq-vae/model/Qwen3-1-7B"
SAVE_DIR="./model/Qwen3-1-7B-expanded-vocab"
INDEX_FILE="../Test/Beauty/Beauty.index.json"

# Create model directory if it doesn't exist
mkdir -p ./model

echo "Base model: $BASE_MODEL"
echo "Save directory: $SAVE_DIR"
echo "Index file: $INDEX_FILE"

# Check if base model exists
if [ ! -d "$BASE_MODEL" ]; then
    echo "Error: Base model directory not found at $BASE_MODEL"
    echo "Please verify the model path is correct for your environment"
    exit 1
fi

# Run the expansion script
python3 expand_qwen_vocab.py

echo "Vocabulary expansion completed!"
echo "Expanded model saved to: $SAVE_DIR"
