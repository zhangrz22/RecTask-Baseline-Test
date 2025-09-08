#!/usr/bin/env python3
"""
Prepare training dataset for SID-to-Title mapping
Combines Beauty.index.json and Beauty.item.json to create training samples
"""

import json
import pandas as pd
import random
import os
from collections import defaultdict
import argparse

def load_json_data(index_path, item_path):
    """
    Load data from Beauty.index.json and Beauty.item.json
    
    Returns:
        tuple: (index_data, item_data)
    """
    print(f"Loading index data from: {index_path}")
    with open(index_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    print(f"Loading item data from: {item_path}")  
    with open(item_path, 'r', encoding='utf-8') as f:
        item_data = json.load(f)
    
    print(f"Loaded {len(index_data)} items from index file")
    print(f"Loaded {len(item_data)} items from item file")
    
    return index_data, item_data

def create_sid_sequence(sid_tokens):
    """
    Convert token list to SID sequence string
    
    Args:
        sid_tokens: List of tokens like ["<|sid_begin|>", "<s_a_156>", ...]
        
    Returns:
        str: Concatenated SID sequence
    """
    return ''.join(sid_tokens)

def create_training_samples(index_data, item_data, prompt_templates):
    """
    Create training samples by matching SID sequences with titles
    
    Args:
        index_data: Dict mapping item_id to SID token list
        item_data: Dict mapping item_id to item info (title, description, etc.)
        prompt_templates: List of prompt templates to use
        
    Returns:
        list: List of training samples
    """
    training_samples = []
    matched_items = 0
    missing_items = 0
    
    for item_id in index_data.keys():
        # Check if item exists in both datasets
        if item_id not in item_data:
            missing_items += 1
            continue
            
        # Get SID sequence
        sid_tokens = index_data[item_id]
        sid_sequence = create_sid_sequence(sid_tokens)
        
        # Get item title and description
        item_info = item_data[item_id]
        title = item_info.get('title', '').strip()
        description = item_info.get('description', '').strip()
        
        # Skip if no title
        if not title:
            continue
            
        # Create training samples with different prompt templates
        for template in prompt_templates:
            if '{sid}' in template and '{title}' in template:
                sample_text = template.format(sid=sid_sequence, title=title)
            elif '{sid}' in template and '{description}' in template and description:
                sample_text = template.format(sid=sid_sequence, description=description)
            elif '{sid}' in template and '{title}' in template and '{description}' in template and description:
                sample_text = template.format(sid=sid_sequence, title=title, description=description)
            else:
                continue
                
            training_samples.append({
                'item_id': item_id,
                'sid_sequence': sid_sequence,
                'title': title,
                'description': description[:500] if description else '',  # Truncate long descriptions
                'text': sample_text
            })
            
        matched_items += 1
    
    print(f"Successfully matched {matched_items} items")
    print(f"Missing items in item dataset: {missing_items}")
    print(f"Total training samples created: {len(training_samples)}")
    
    return training_samples

def get_prompt_templates():
    """
    Define various prompt templates for training
    
    Returns:
        list: List of prompt templates
    """
    templates = [
        # Simple mapping format
        "{sid} represents {title}",
        "{sid} is {title}",
        "Product {sid}: {title}",
        "Item {sid} = {title}",
        
        # Question-answer format  
        "What is {sid}? It is {title}",
        "Q: What does {sid} represent? A: {title}",
        "Question: What is {sid}? Answer: {title}",
        
        # Natural language format
        "The product represented by {sid} is {title}",
        "When you see {sid}, it refers to {title}",
        "{sid} corresponds to the product: {title}",
        
        # Instruction format
        "Identify this product: {sid}\nAnswer: {title}",
        "Product identification: {sid} -> {title}",
        
        # Conversation format
        "User: What is {sid}?\nAssistant: {sid} represents {title}",
        "Human: Can you tell me what {sid} is?\nAssistant: {sid} is {title}",
    ]
    
    return templates

def split_dataset(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train/validation/test sets
    
    Args:
        samples: List of training samples
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set  
        test_ratio: Ratio for test set
        
    Returns:
        tuple: (train_samples, val_samples, test_samples)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Shuffle samples
    random.shuffle(samples)
    
    total_samples = len(samples)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    train_samples = samples[:train_size]
    val_samples = samples[train_size:train_size + val_size]
    test_samples = samples[train_size + val_size:]
    
    print(f"Dataset split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    return train_samples, val_samples, test_samples

def save_dataset(samples, output_path, format='parquet'):
    """
    Save dataset to file
    
    Args:
        samples: List of samples
        output_path: Output file path
        format: Output format ('parquet', 'json', 'jsonl')
    """
    if format == 'parquet':
        df = pd.DataFrame(samples)
        df.to_parquet(output_path, index=False)
    elif format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
    elif format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(samples)} samples to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Prepare SID-to-Title training dataset')
    parser.add_argument('--index_file', default='../Test/Beauty/Beauty.index.json', 
                       help='Path to Beauty.index.json')
    parser.add_argument('--item_file', default='../Test/Beauty/Beauty.item.json',
                       help='Path to Beauty.item.json') 
    parser.add_argument('--output_dir', default='./data',
                       help='Output directory for processed datasets')
    parser.add_argument('--format', choices=['parquet', 'json', 'jsonl'], default='parquet',
                       help='Output format')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    try:
        index_data, item_data = load_json_data(args.index_file, args.item_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the file paths are correct")
        return
    
    # Get prompt templates
    templates = get_prompt_templates()
    print(f"Using {len(templates)} prompt templates")
    
    # Create training samples
    samples = create_training_samples(index_data, item_data, templates)
    
    if not samples:
        print("No training samples created. Please check your data files.")
        return
    
    # Split dataset
    train_samples, val_samples, test_samples = split_dataset(samples)
    
    # Save datasets
    file_extension = 'parquet' if args.format == 'parquet' else 'json' if args.format == 'json' else 'jsonl'
    
    save_dataset(train_samples, 
                os.path.join(args.output_dir, f'train.{file_extension}'), 
                args.format)
    save_dataset(val_samples, 
                os.path.join(args.output_dir, f'val.{file_extension}'), 
                args.format)
    save_dataset(test_samples, 
                os.path.join(args.output_dir, f'test.{file_extension}'), 
                args.format)
    
    # Print sample examples
    print("\n=== Sample Training Examples ===")
    for i, sample in enumerate(random.sample(train_samples, min(5, len(train_samples)))):
        print(f"\nExample {i+1}:")
        print(f"SID: {sample['sid_sequence']}")
        print(f"Title: {sample['title'][:100]}...")
        print(f"Training text: {sample['text'][:150]}...")
    
    print(f"\nDataset preparation completed!")
    print(f"Files saved in: {args.output_dir}")

if __name__ == "__main__":
    main()