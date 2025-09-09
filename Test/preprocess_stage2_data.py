#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理Stage2验证数据，避免测试时重复处理
"""
import pandas as pd
import json
import argparse
import os
from tqdm import tqdm

def preprocess_stage2_data(input_path, output_path, sample_num=-1):
    """预处理Stage2数据并保存为更高效的格式"""
    print(f"🔄 Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    if sample_num > 0:
        df = df.head(sample_num)
        print(f"📊 Using {sample_num} samples")
    
    processed_data = []
    all_items = set()
    
    print("🚀 Processing data...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # 复制Stage2ValDataset的处理逻辑
        if 'text' in row and pd.notna(row['text']):
            full_text = row['text']
            if '### ' in full_text:
                system_instruction = full_text.split('### ')[0].strip()
                instruction_part = full_text.split('### ')[1].strip()
                if '### ' in instruction_part:
                    instruction_part = instruction_part.split('### ')[0].strip()
                full_prompt = system_instruction + "\n\n" + instruction_part
            else:
                full_prompt = row['instruction']
        else:
            instruction = row['instruction']
            full_prompt = f"You are an expert in Recommender System. The user has interacted with several items in chronological order. Can you predict the next possible item that the user may expect?\n\n{instruction}"
        
        response = row['response']
        all_items.add(response)
        
        processed_data.append({
            'input_ids': full_prompt,
            'labels': response
        })
    
    # 保存处理后的数据
    print(f"💾 Saving processed data to {output_path}...")
    output_data = {
        'data': processed_data,
        'all_items': list(all_items),
        'total_samples': len(processed_data)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Processed {len(processed_data)} samples")
    print(f"📈 Total unique items: {len(all_items)}")
    return output_data

class PreprocessedStage2Dataset:
    """使用预处理数据的快速数据集"""
    def __init__(self, preprocessed_path):
        print(f"⚡ Loading preprocessed data from {preprocessed_path}...")
        with open(preprocessed_path, 'r', encoding='utf-8') as f:
            self.data_dict = json.load(f)
        
        self.data = self.data_dict['data']
        self.all_items_cache = self.data_dict['all_items']
        print(f"✅ Loaded {len(self.data)} preprocessed samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_all_items(self):
        return self.all_items_cache
    
    def get_prefix_allowed_tokens_fn(self, tokenizer):
        # 复制原始的约束逻辑
        allowed_tokens = {}
        
        # 位置0: <|sid_begin|>
        sid_begin_id = tokenizer.convert_tokens_to_ids('<|sid_begin|>')
        allowed_tokens[0] = {sid_begin_id} if sid_begin_id != tokenizer.unk_token_id else set()
        
        # 位置1-4: s_a, s_b, s_c, s_d tokens
        for i, prefix in enumerate(['s_a', 's_b', 's_c', 's_d'], 1):
            allowed_tokens[i] = set()
            for j in range(256):
                token = f'<{prefix}_{j}>'
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    allowed_tokens[i].add(token_id)
        
        # 位置5: <|sid_end|>
        sid_end_id = tokenizer.convert_tokens_to_ids('<|sid_end|>')
        allowed_tokens[5] = {sid_end_id} if sid_end_id != tokenizer.unk_token_id else set()
        
        # 位置6+: EOS
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        allowed_tokens[6] = {eos_id}
        
        sep = tokenizer("Response:", add_special_tokens=False)["input_ids"]
        
        def find_last_sublist(lst, sub):
            if not sub:
                return None
            n, m = len(lst), len(sub)
            for start in range(n - m, -1, -1):
                if lst[start:start + m] == sub:
                    return start
            return None
        
        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            pos = find_last_sublist(sentence, sep)
            if pos is None:
                try:
                    vocab_size = getattr(tokenizer, 'vocab_size', None) or len(tokenizer)
                except Exception:
                    vocab_size = 50257
                return list(range(vocab_size))
            
            gen_pos = len(sentence) - (pos + len(sep))
            if gen_pos in allowed_tokens:
                return list(allowed_tokens[gen_pos])
            else:
                return list(allowed_tokens[6])
        
        return prefix_allowed_tokens_fn

def main():
    parser = argparse.ArgumentParser(description="Preprocess Stage2 validation data")
    parser.add_argument("--input_path", type=str, 
                        default="../Qwen3/data_stage2/val.parquet",
                        help="Input parquet file path")
    parser.add_argument("--output_path", type=str,
                        default="../Qwen3/data_stage2/val_preprocessed.json",
                        help="Output preprocessed JSON file path")
    parser.add_argument("--sample_num", type=int, default=-1,
                        help="Number of samples to process (-1 for all)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 预处理数据
    preprocess_stage2_data(args.input_path, args.output_path, args.sample_num)

if __name__ == "__main__":
    main()