#!/usr/bin/env python3
"""
Script to expand Qwen3 model vocabulary with recommendation-specific tokens
Adds special tokens: <|sid_begin|>, <|sid_end|>, and s_a_*, s_b_*, s_c_*, s_d_* tokens
"""

import os
import json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

def load_special_tokens_from_index(index_file_path):
    """
    Load special tokens from Beauty.index.json file
    
    Args:
        index_file_path: Path to Beauty.index.json file
        
    Returns:
        set: Set of unique special tokens found in the index file
    """
    if not os.path.exists(index_file_path):
        print(f"Warning: Index file {index_file_path} not found. Using default token ranges.")
        return set()
    
    special_tokens = set()
    
    try:
        with open(index_file_path, 'r') as f:
            index_data = json.load(f)
        
        # Extract all unique tokens from the index
        for item_id, tokens in index_data.items():
            for token in tokens:
                special_tokens.add(token)
                
        print(f"Found {len(special_tokens)} unique tokens in index file")
        return special_tokens
        
    except Exception as e:
        print(f"Error reading index file: {e}")
        return set()

def create_token_list(index_file_path, max_range=256):
    """
    Create comprehensive token list for recommendation system
    
    Args:
        index_file_path: Path to Beauty.index.json file
        max_range: Maximum range for s_* tokens (default: 256)
        
    Returns:
        list: List of tokens to add to vocabulary
    """
    # Load tokens from index file
    index_tokens = load_special_tokens_from_index(index_file_path)
    
    # Create comprehensive token set
    all_tokens = set()
    
    # Add control tokens
    all_tokens.add('<|sid_begin|>')
    all_tokens.add('<|sid_end|>')
    
    # Add s_* tokens based on index file and extended range
    for prefix in ['s_a', 's_b', 's_c', 's_d']:
        for i in range(max_range):
            all_tokens.add(f'<{prefix}_{i}>')
    
    # Add any additional tokens found in index file
    all_tokens.update(index_tokens)
    
    return sorted(list(all_tokens))

def expand_qwen_vocabulary(base_model_path, save_dir, index_file_path, max_token_range=256):
    """
    Expand Qwen3 model vocabulary with recommendation-specific tokens
    
    Args:
        base_model_path: Path to base Qwen3 model
        save_dir: Directory to save expanded model
        index_file_path: Path to Beauty.index.json file
        max_token_range: Maximum range for s_* tokens
    """
    
    print("Starting Qwen3 vocabulary expansion...")
    print(f"Base model: {base_model_path}")
    print(f"Save directory: {save_dir}")
    print(f"Index file: {index_file_path}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load original model components
    print("Loading original model components...")
    config = AutoConfig.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    print(f"Original vocabulary size: {len(tokenizer)}")
    print(f"Original config vocab_size: {config.vocab_size}")
    
    # 2. Create token list
    print("Creating recommendation token list...")
    add_tokens = create_token_list(index_file_path, max_token_range)
    
    print(f"Tokens to add: {len(add_tokens)}")
    print(f"Sample tokens: {add_tokens[:10]}")
    
    # 3. Add new tokens to tokenizer
    print("Adding tokens to tokenizer...")
    num_added = tokenizer.add_tokens(add_tokens)
    print(f"Successfully added {num_added} new tokens")
    
    # 4. Calculate target vocabulary size (aligned to 256)
    current_vocab_size = len(tokenizer)
    target_vocab_size = (current_vocab_size + 255) // 256 * 256  # Round up to nearest 256
    
    print(f"Current vocabulary size after adding tokens: {current_vocab_size}")
    print(f"Target vocabulary size (aligned): {target_vocab_size}")
    
    # 5. Resize model embeddings
    print("Resizing model token embeddings...")
    model.resize_token_embeddings(target_vocab_size)
    
    # 6. Update configuration
    config.vocab_size = target_vocab_size
    
    # 7. Save all components
    print("Saving expanded model components...")
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    config.save_pretrained(save_dir)
    
    print(f"Model expansion completed! Saved to: {save_dir}")
    
    # 8. Test the expanded model
    print("Testing expanded model...")
    test_expanded_model(save_dir, add_tokens[:5])

def test_expanded_model(model_dir, sample_tokens):
    """
    Test the expanded model with sample tokens
    
    Args:
        model_dir: Path to expanded model directory
        sample_tokens: Sample tokens to test
    """
    try:
        # Load expanded model
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)
        
        # Test tokenization
        test_text = f"{' '.join(sample_tokens[:3])} Hello world"
        print(f"Test input: {test_text}")
        
        # Tokenize
        input_ids = tokenizer.encode(test_text, return_tensors='pt')
        print(f"Input IDs: {input_ids}")
        print(f"Tokenized: {[tokenizer.decode([id]) for id in input_ids[0]]}")
        
        # Test generation (if CUDA available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            model = model.to(device)
            input_ids = input_ids.to(device)
            
            print("Testing text generation...")
            with torch.no_grad():
                output = model.generate(
                    input_ids, 
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
            print(f"Generated: {generated_text}")
        else:
            print("CUDA not available, skipping generation test")
            
        print("Model test completed successfully!")
        
    except Exception as e:
        print(f"Error during model testing: {e}")

def main():
    # Configuration - adjust paths as needed for cloud environment
    base_model_path = "/home/liuzhanyu/SidTextAlign/content-rq-vae/model/Qwen3-1-7B"
    save_dir = "./model/Qwen3-1-7B-expanded-vocab"
    
    # Paths relative to current working directory (should work on cloud)
    index_file_path = "../Test/Beauty/Beauty.index.json"  # Relative path from Qwen3 folder
    
    # Check if base model exists
    if not os.path.exists(base_model_path):
        print(f"Error: Base model not found at {base_model_path}")
        print("Please verify the model path is correct for your environment")
        return
    
    # Check if index file exists (use absolute path if provided)
    if not os.path.exists(index_file_path):
        # Try alternative path
        alt_index_path = "/Users/zrz/Desktop/Rec_test/Test/Beauty/Beauty.index.json"
        if os.path.exists(alt_index_path):
            index_file_path = alt_index_path
        else:
            print(f"Warning: Index file not found at {index_file_path}")
            print("Proceeding with default token ranges...")
    
    # Expand vocabulary
    expand_qwen_vocabulary(
        base_model_path=base_model_path,
        save_dir=save_dir,
        index_file_path=index_file_path,
        max_token_range=256
    )

if __name__ == "__main__":
    main()