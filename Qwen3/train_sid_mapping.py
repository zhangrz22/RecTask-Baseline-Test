#!/usr/bin/env python3
"""
Fine-tuning script for SID-to-Title mapping using Qwen3 with expanded vocabulary
Uses LoRA and TrainableTokensConfig for efficient training of special tokens
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
from peft import LoraConfig, get_peft_model, TaskType, TrainableTokensConfig
import argparse

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: Optional[str] = field(
        default="./model/Qwen3-1-7B-expanded-vocab",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA fine-tuning"}
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=128, 
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )

@dataclass 
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: str = field(
        default="./data",
        metadata={"help": "Directory containing the processed datasets"}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length"}
    )
    train_file: Optional[str] = field(
        default="train.parquet",
        metadata={"help": "Training data file"}
    )
    validation_file: Optional[str] = field(
        default="val.parquet", 
        metadata={"help": "Validation data file"}
    )

def load_dataset_from_file(file_path, data_format='parquet'):
    """
    Load dataset from file
    
    Args:
        file_path: Path to dataset file
        data_format: File format ('parquet', 'json', 'jsonl')
        
    Returns:
        Dataset: HuggingFace dataset
    """
    if data_format == 'parquet':
        df = pd.read_parquet(file_path)
        return Dataset.from_pandas(df)
    elif data_format == 'json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_list(data)
    elif data_format == 'jsonl':
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return Dataset.from_list(data)
    else:
        raise ValueError(f"Unsupported format: {data_format}")

def get_special_token_ids(tokenizer, max_range=256):
    """
    Get token IDs for special recommendation tokens
    
    Args:
        tokenizer: Tokenizer instance
        max_range: Maximum range for s_* tokens
        
    Returns:
        list: List of special token IDs
    """
    special_tokens = []
    
    # Add control tokens
    special_tokens.extend(['<|sid_begin|>', '<|sid_end|>'])
    
    # Add s_* tokens
    for prefix in ['s_a', 's_b', 's_c', 's_d']:
        for i in range(max_range):
            special_tokens.append(f'<{prefix}_{i}>')
    
    # Convert to token IDs
    special_token_ids = []
    for token in special_tokens:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:  # Valid token
                special_token_ids.append(token_id)
        except:
            continue
            
    print(f"Found {len(special_token_ids)} special tokens in vocabulary")
    if special_token_ids:
        print(f"Special token ID range: {min(special_token_ids)} - {max(special_token_ids)}")
    
    return special_token_ids

def tokenize_function(examples, tokenizer, max_length):
    """
    Tokenize examples for training
    
    Args:
        examples: Batch of examples
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        dict: Tokenized examples with labels
    """
    # Tokenize the text
    tokenized = tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt" if len(examples['text']) == 1 else None,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    
    # For causal language modeling, labels are the same as input_ids
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized

def setup_model_and_tokenizer(model_args):
    """
    Setup model and tokenizer with LoRA configuration
    
    Args:
        model_args: Model arguments
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model from: {model_args.model_name_or_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    if model_args.use_lora:
        print("Setting up LoRA configuration...")
        
        # Get special token IDs for TrainableTokensConfig
        special_token_ids = get_special_token_ids(tokenizer)
        
        if special_token_ids:
            # Use TrainableTokensConfig to train special tokens
            lora_config = TrainableTokensConfig(
                token_indices=special_token_ids,
                target_modules=["embed_tokens", "lm_head"],  # Train both input and output embeddings
                init_weights=True
            )
        else:
            # Fallback to regular LoRA
            print("No special tokens found, using regular LoRA")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=model_args.lora_rank,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
            )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Print trainable parameters for debugging
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Trainable parameter: {name}")
    
    return model, tokenizer

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set label names for trainer
    training_args.label_names = ["labels"]
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # Load datasets
    print(f"Loading datasets from: {data_args.data_dir}")
    
    train_file_path = os.path.join(data_args.data_dir, data_args.train_file)
    val_file_path = os.path.join(data_args.data_dir, data_args.validation_file)
    
    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"Training file not found: {train_file_path}")
    
    # Determine file format from extension
    file_ext = data_args.train_file.split('.')[-1]
    data_format = 'parquet' if file_ext == 'parquet' else 'jsonl' if file_ext == 'jsonl' else 'json'
    
    # Load training dataset
    train_dataset = load_dataset_from_file(train_file_path, data_format)
    print(f"Loaded training dataset: {len(train_dataset)} samples")
    
    # Load validation dataset if exists
    eval_dataset = None
    if os.path.exists(val_file_path):
        eval_dataset = load_dataset_from_file(val_file_path, data_format)
        print(f"Loaded validation dataset: {len(eval_dataset)} samples")
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, data_args.max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data"
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda x: tokenize_function(x, tokenizer, data_args.max_seq_length),
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing validation data"
        )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate
    if eval_dataset:
        print("Running final evaluation...")
        eval_result = trainer.evaluate()
        print(f"Final evaluation result: {eval_result}")
    
    # Save model
    print(f"Saving model to: {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print("Training completed!")

if __name__ == "__main__":
    main()