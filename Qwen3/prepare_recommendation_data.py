#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第二阶段推荐训练数据准备
基于用户交互序列生成推荐训练数据，通过移位方式生成多条样本
"""

import json
import pandas as pd
import argparse
import os
from typing import Dict, List, Tuple
import random

# SFT prompt模板
SFT_PROMPT = "You are an expert in Recommender System. The user has interacted with several items in chronological order. Can you predict the next possible item that the user may expect?" \
             "\n\n### {instruction}\n\n### {response}"

PROMPT_TEMPLATE = {
    "instruction": "The user's previous interaction history is as follows: {inters}",
    "response": "{item}"
}

def load_interaction_data(inter_file: str) -> Dict[str, List[int]]:
    """
    加载用户交互数据
    
    Args:
        inter_file: 交互文件路径
        
    Returns:
        Dict[user_id, List[item_ids]]
    """
    print(f"Loading interaction data from: {inter_file}")
    
    with open(inter_file, 'r', encoding='utf-8') as f:
        interactions = json.load(f)
    
    print(f"Loaded {len(interactions)} users")
    return interactions

def load_index_data(index_file: str) -> Dict[str, List[str]]:
    """
    加载物品索引数据（item_id -> SID tokens）
    
    Args:
        index_file: 索引文件路径
        
    Returns:
        Dict[item_id, List[sid_tokens]]
    """
    print(f"Loading index data from: {index_file}")
    
    with open(index_file, 'r', encoding='utf-8') as f:
        indices = json.load(f)
    
    print(f"Loaded {len(indices)} items")
    return indices

def convert_items_to_sids(item_ids: List[int], indices: Dict[str, List[str]]) -> List[str]:
    """
    将item_ids转换为SID字符串
    
    Args:
        item_ids: 物品ID列表
        indices: 物品索引映射
        
    Returns:
        List[sid_strings]
    """
    sids = []
    for item_id in item_ids:
        item_key = str(item_id)
        if item_key in indices:
            sid_tokens = indices[item_key]
            sid_string = "".join(sid_tokens)
            sids.append(sid_string)
        else:
            print(f"Warning: item {item_id} not found in index")
    
    return sids

def generate_training_samples(
    user_interactions: Dict[str, List[int]], 
    indices: Dict[str, List[str]],
    min_seq_len: int = 3,
    his_sep: str = ", "
) -> List[Dict[str, str]]:
    """
    生成训练样本
    
    Args:
        user_interactions: 用户交互数据
        indices: 物品索引数据
        min_seq_len: 最小序列长度
        his_sep: 历史序列分隔符
        
    Returns:
        List[training_samples]
    """
    training_samples = []
    total_users = len(user_interactions)
    processed_users = 0
    
    print(f"Generating training samples for {total_users} users...")
    
    for user_id, item_sequence in user_interactions.items():
        processed_users += 1
        
        if len(item_sequence) < min_seq_len + 1:
            continue  # 序列太短，跳过
        
        # 转换为SID
        sid_sequence = convert_items_to_sids(item_sequence, indices)
        if len(sid_sequence) != len(item_sequence):
            continue  # 某些item没有对应的SID，跳过
        
        # 生成训练样本：至少min_seq_len个历史，预测下一个
        # 但留最后一个不用于训练（用于验证）
        max_train_len = len(sid_sequence) - 1  # 留最后一个做验证
        
        for end_idx in range(min_seq_len + 1, max_train_len + 1):
            # 历史序列：前end_idx-1个
            history_sids = sid_sequence[:end_idx-1]
            # 目标：第end_idx个
            target_sid = sid_sequence[end_idx-1]
            
            # 构建输入文本
            history_text = his_sep.join(history_sids)
            instruction = PROMPT_TEMPLATE["instruction"].format(inters=history_text)
            response = PROMPT_TEMPLATE["response"].format(item=target_sid)
            
            # 构建完整的SFT格式
            full_text = SFT_PROMPT.format(instruction=instruction, response=response)
            
            sample = {
                "user_id": user_id,
                "history_length": len(history_sids),
                "instruction": instruction,
                "response": response,
                "text": full_text
            }
            
            training_samples.append(sample)
        
        if processed_users % 1000 == 0:
            print(f"Processed {processed_users}/{total_users} users, generated {len(training_samples)} samples")
    
    print(f"Generated {len(training_samples)} training samples from {processed_users} users")
    return training_samples

def generate_validation_samples(
    user_interactions: Dict[str, List[int]], 
    indices: Dict[str, List[str]],
    min_seq_len: int = 3,
    his_sep: str = ", "
) -> List[Dict[str, str]]:
    """
    生成验证样本（使用每个用户序列的最后一个作为验证）
    
    Args:
        user_interactions: 用户交互数据
        indices: 物品索引数据
        min_seq_len: 最小序列长度
        his_sep: 历史序列分隔符
        
    Returns:
        List[validation_samples]
    """
    validation_samples = []
    
    print("Generating validation samples...")
    
    for user_id, item_sequence in user_interactions.items():
        if len(item_sequence) < min_seq_len + 1:
            continue
            
        # 转换为SID
        sid_sequence = convert_items_to_sids(item_sequence, indices)
        if len(sid_sequence) != len(item_sequence):
            continue
        
        # 使用倒数第二个之前的作为历史，最后一个作为目标
        history_sids = sid_sequence[:-1]
        target_sid = sid_sequence[-1]
        
        # 构建输入文本
        history_text = his_sep.join(history_sids)
        instruction = PROMPT_TEMPLATE["instruction"].format(inters=history_text)
        response = PROMPT_TEMPLATE["response"].format(item=target_sid)
        
        # 构建完整的SFT格式
        full_text = SFT_PROMPT.format(instruction=instruction, response=response)
        
        sample = {
            "user_id": user_id,
            "history_length": len(history_sids),
            "instruction": instruction,
            "response": response,
            "text": full_text
        }
        
        validation_samples.append(sample)
    
    print(f"Generated {len(validation_samples)} validation samples")
    return validation_samples

def save_samples_to_parquet(samples: List[Dict], output_file: str):
    """
    保存样本到parquet文件
    
    Args:
        samples: 训练样本列表
        output_file: 输出文件路径
    """
    df = pd.DataFrame(samples)
    print(f"Saving {len(samples)} samples to {output_file}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample text preview: {df['text'].iloc[0][:200]}...")
    
    df.to_parquet(output_file, index=False)
    print(f"Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Prepare recommendation training data")
    parser.add_argument("--data_path", type=str, default="../Test/Beauty", 
                        help="Path to data directory")
    parser.add_argument("--inter_file", type=str, default="Beauty.inter.json",
                        help="Interaction file name")
    parser.add_argument("--index_file", type=str, default="Beauty.index.json", 
                        help="Index file name")
    parser.add_argument("--output_dir", type=str, default="./data_stage2",
                        help="Output directory")
    parser.add_argument("--min_seq_len", type=int, default=3,
                        help="Minimum sequence length")
    parser.add_argument("--his_sep", type=str, default=", ",
                        help="History separator")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建文件路径
    inter_file = os.path.join(args.data_path, args.inter_file)
    index_file = os.path.join(args.data_path, args.index_file)
    
    # 加载数据
    user_interactions = load_interaction_data(inter_file)
    indices = load_index_data(index_file)
    
    # 生成训练样本
    training_samples = generate_training_samples(
        user_interactions, indices, args.min_seq_len, args.his_sep
    )
    
    # 生成验证样本
    validation_samples = generate_validation_samples(
        user_interactions, indices, args.min_seq_len, args.his_sep
    )
    
    print(f"\nData split summary:")
    print(f"  Training samples: {len(training_samples)} (移位生成)")
    print(f"  Validation/Test samples: {len(validation_samples)} (用户最后交互)")
    
    # 保存数据
    train_file = os.path.join(args.output_dir, "train.parquet")
    val_file = os.path.join(args.output_dir, "val.parquet")
    
    save_samples_to_parquet(training_samples, train_file)
    save_samples_to_parquet(validation_samples, val_file)  # 验证和测试用同一份
    
    # 保存配置信息
    config = {
        "data_path": args.data_path,
        "inter_file": args.inter_file,
        "index_file": args.index_file,
        "min_seq_len": args.min_seq_len,
        "his_sep": args.his_sep,
        "train_samples": len(training_samples),
        "val_test_samples": len(validation_samples),
        "total_users": len(user_interactions),
        "total_items": len(indices)
    }
    
    config_file = os.path.join(args.output_dir, "data_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\nData preparation completed!")
    print(f"Files saved to: {args.output_dir}")
    print(f"Config saved to: {config_file}")

if __name__ == "__main__":
    main()