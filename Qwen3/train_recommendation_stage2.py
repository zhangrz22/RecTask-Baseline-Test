#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第二阶段推荐训练脚本
在第一阶段SID映射训练基础上，进一步加强推荐能力
使用LoRA训练更多参数（Q,K,V,O等）
"""

from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, 
    Trainer, TrainingArguments, DataCollatorForLanguageModeling, 
    HfArgumentParser
)
import os
import torch
import json
import pandas as pd
from datasets import Dataset, load_dataset
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import random
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import argparse

@dataclass
class ModelArguments:
    """
    模型相关参数
    """
    base_model_path: Optional[str] = field(
        default="../Qwen3/model/Qwen3-1-7B-expanded-vocab",
        metadata={"help": "第一阶段基础模型路径"}
    )
    stage1_lora_path: Optional[str] = field(
        default="../Qwen3/results/sid_mapping_model", 
        metadata={"help": "第一阶段LoRA模型路径"}
    )
    max_token_range: int = field(
        default=256,
        metadata={"help": "扩展的特殊token范围（s_a_0到s_a_255等）"}
    )
    # LoRA参数
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "LoRA目标模块，逗号分隔"}
    )

@dataclass 
class DataArguments:
    """
    数据相关参数
    """
    data_dir: str = field(
        default="./data_stage2",
        metadata={"help": "第二阶段训练数据目录"}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "最大序列长度"}
    )
    train_file: Optional[str] = field(
        default="train.parquet",
        metadata={"help": "训练数据文件"}
    )
    validation_file: Optional[str] = field(
        default="val.parquet", 
        metadata={"help": "验证数据文件"}
    )

def load_dataset_from_file(file_path, data_format='parquet'):
    """
    加载数据集
    """
    if data_format == 'parquet':
        df = pd.read_parquet(file_path)
        return Dataset.from_pandas(df)
    elif data_format == 'json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_list(data)
    else:
        raise ValueError(f"Unsupported format: {data_format}")

def get_extended_special_tokens(max_range=256):
    """
    获取扩展的特殊token列表（与第一阶段保持一致）
    
    Args:
        max_range: token范围（默认256）
        
    Returns:
        list: 特殊token列表
    """
    special_tokens = []
    
    # 添加控制token
    special_tokens.extend(['<|sid_begin|>', '<|sid_end|>'])
    
    # 添加s_*类型的token（4组×256）
    for prefix in ['s_a', 's_b', 's_c', 's_d']:
        for i in range(max_range):
            special_tokens.append(f'<{prefix}_{i}>')
    
    return special_tokens

def tokenize_function(examples, tokenizer, max_length=1024):
    """
    数据tokenization函数
    """
    tokenized = tokenizer(
        examples['text'],
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt" if len(examples['text']) == 1 else None,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    
    # 对于因果语言模型，labels与input_ids相同
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized

def setup_model_and_tokenizer(model_args):
    """
    设置模型和tokenizer
    
    Args:
        model_args: 模型参数
        
    Returns:
        tuple: (model, tokenizer)
    """
    print("="*60)
    print("🔄 Setting up Stage 2 model...")
    
    # 1. 加载tokenizer（从第一阶段LoRA模型路径，包含扩展的词汇表）
    print("📝 Loading tokenizer from Stage 1 LoRA model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_args.stage1_lora_path)
        print(f"✅ Tokenizer loaded from LoRA path: {model_args.stage1_lora_path}")
    except Exception as e:
        print(f"⚠️ Failed to load tokenizer from LoRA path: {e}")
        print("📝 Falling back to base model tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_args.base_model_path, use_fast=False)
        print(f"✅ Tokenizer loaded from base model with slow tokenizer")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Tokenizer loaded, vocab size: {len(tokenizer)}")
    
    # 2. 加载第一阶段基础模型
    print("🏗️ Loading stage 1 base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.base_model_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )
    
    # 3. 加载第一阶段LoRA权重
    print("🔧 Loading stage 1 LoRA weights...")
    stage1_model = PeftModel.from_pretrained(base_model, model_args.stage1_lora_path)
    
    # 4. 合并第一阶段权重
    print("🔀 Merging stage 1 weights...")
    model = stage1_model.merge_and_unload()
    
    print("✅ Stage 1 model loaded and merged")
    
    # 5. 准备第二阶段LoRA配置
    print("⚙️ Setting up stage 2 LoRA config...")
    
    # 获取扩展的特殊token IDs
    special_tokens = get_extended_special_tokens(model_args.max_token_range)
    tokenized_special_tokens = tokenizer.convert_tokens_to_ids(special_tokens)
    
    # 过滤掉不存在的token
    valid_special_tokens = [tid for tid in tokenized_special_tokens if tid != tokenizer.unk_token_id]
    
    print(f"Special tokens: {len(special_tokens)} defined")
    print(f"Valid special token IDs: {len(valid_special_tokens)}")
    print(f"Example special token IDs: {valid_special_tokens[:10]}")
    
    # LoRA目标模块
    target_modules = model_args.lora_target_modules.split(',')
    target_modules = [mod.strip() for mod in target_modules]
    
    print(f"Target modules: {target_modules}")
    print(f"Embedding layer size: {model.get_input_embeddings().weight.shape[0]}")
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # 指定需要训练的特殊token
        trainable_token_indices={
            'embed_tokens': valid_special_tokens,
        }
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 打印可训练参数信息
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if len(list(model.named_parameters())) < 20:  # 只在参数不多时打印
                print(f"Trainable: {name} - {param.shape}")
    
    print(f"\n📊 Parameter Summary:")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  All params: {all_params:,}")
    print(f"  Trainable %: {100 * trainable_params / all_params:.2f}%")
    
    print("="*60)
    
    return model, tokenizer

def main():
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置label names
    training_args.label_names = ["labels"]
    
    # 确保输出目录存在
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs(training_args.logging_dir, exist_ok=True)
    
    print(f"Output directory: {os.path.abspath(training_args.output_dir)}")
    print(f"Logging directory: {os.path.abspath(training_args.logging_dir)}")
    
    # 设置模型和tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # 加载数据集
    print("\n📊 Loading datasets...")
    
    train_file = os.path.join(data_args.data_dir, data_args.train_file)
    val_file = os.path.join(data_args.data_dir, data_args.validation_file)
    
    print(f"Loading training data from: {train_file}")
    train_dataset = load_dataset_from_file(train_file)
    
    print(f"Loading validation data from: {val_file}")
    eval_dataset = load_dataset_from_file(val_file)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # 数据预处理
    print("🔄 Tokenizing datasets...")
    
    def tokenize_fn(examples):
        return tokenize_function(examples, tokenizer, data_args.max_seq_length)
    
    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset"
    )
    
    eval_dataset = eval_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval dataset"
    )
    
    print("✅ Tokenization completed")
    
    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言模型
        pad_to_multiple_of=8,
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("\n🚀 Starting Stage 2 Recommendation Training...")
    print(f"Training arguments: {training_args}")
    
    trainer.train()
    
    # 保存模型
    print("💾 Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # 保存训练配置
    config_dict = {
        "stage": 2,
        "base_model_path": model_args.base_model_path,
        "stage1_lora_path": model_args.stage1_lora_path,
        "max_token_range": model_args.max_token_range,
        "lora_r": model_args.lora_r,
        "lora_alpha": model_args.lora_alpha,
        "lora_dropout": model_args.lora_dropout,
        "lora_target_modules": model_args.lora_target_modules,
        "max_seq_length": data_args.max_seq_length,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
    }
    
    config_file = os.path.join(training_args.output_dir, "stage2_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Stage 2 training completed!")
    print(f"Model saved to: {training_args.output_dir}")
    print(f"Config saved to: {config_file}")

if __name__ == "__main__":
    main()