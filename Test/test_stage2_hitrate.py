#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 Stage 2 推荐模型的Hit Rate测试脚本
测试Stage 2训练后的推荐效果，并与Stage 1进行对比
"""
import argparse
import json
import os
import sys
import torch
import transformers
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import SeqRecDataset
from utils import *
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results
import logging
import random
import datetime
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from torch.utils.data import ConcatDataset
import pandas as pd

class Stage2ValDataset:
    """用于Stage 2验证数据的数据集类"""
    def __init__(self, val_data_path, sample_num=-1):
        # 加载验证数据
        self.df = pd.read_parquet(val_data_path)
        
        if sample_num > 0:
            self.df = self.df.head(sample_num)
        
        self.data = []
        for _, row in self.df.iterrows():
            # 从instruction中提取历史序列
            instruction = row['instruction']
            # 从response中提取目标
            response = row['response']
            
            # 匹配TestCollator期望的格式: input_ids, labels
            self.data.append({
                'input_ids': instruction + "\nResponse:",  # TestCollator会处理这个格式
                'labels': response
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_all_items(self):
        # 返回所有可能的items（仏labels中提取）
        items = set()
        for item in self.data:
            items.add(item['labels'])
        return list(items)
    
    def get_prefix_allowed_tokens_fn(self, tokenizer):
        # 简化版本，允许所有SID tokens
        def prefix_allowed_tokens(batch_id, input_ids):
            # 允许所有特殊tokens
            allowed_tokens = []
            vocab = tokenizer.get_vocab()
            for token, token_id in vocab.items():
                if token.startswith('<s_') or token in ['<|sid_begin|>', '<|sid_end|>']:
                    allowed_tokens.append(token_id)
            return allowed_tokens
        return prefix_allowed_tokens

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 Stage 2 Hit Rate Test")
    
    # 基础参数
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_path", type=str, default="./", help="data directory")
    parser.add_argument("--dataset", type=str, default="Beauty", help="Dataset name")
    parser.add_argument("--index_file", type=str, default=".index.json", help="the item indices file")
    
    # 模型路径参数 - Stage 2 模型
    parser.add_argument("--base_model_path", type=str, 
                        default="../Qwen3/model/Qwen3-1-7B-expanded-vocab",
                        help="Base model path (expanded vocab)")
    parser.add_argument("--stage2_model_path", type=str,
                        default="../Qwen3/results/stage2_recommendation_model", 
                        help="Stage 2 model path")
    
    # Stage 1模型路径（Stage 2需要）
    parser.add_argument("--stage1_model_path", type=str,
                        default="../Qwen3/results/sid_mapping_model", 
                        help="Stage 1 model path (needed for Stage 2 loading)")
    
    # Stage 2验证数据路径（固定使用）
    parser.add_argument("--stage2_val_data_path", type=str,
                        default="../Qwen3/data_stage2/val.parquet", 
                        help="Path to Stage 2 validation data")
    
    # 数据相关参数
    parser.add_argument("--max_his_len", type=int, default=20,
                        help="the max number of items in history sequence")
    parser.add_argument("--add_prefix", action="store_true", default=False,
                        help="whether add sequential prefix in history")
    parser.add_argument("--his_sep", type=str, default=", ", help="The separator used for history")
    
    # 测试参数
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=-1,
                        help="test sample number, -1 represents using all test data")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
                        help="test metrics, separate by comma")
    parser.add_argument("--filter_items", action="store_true", default=False,
                        help="whether filter illegal items")
    
    # CoT/diagnostics
    parser.add_argument("--enable_cot", action="store_true", default=False,
                        help="enable two-stage generation: Think then constrained Response")
    parser.add_argument("--think_max_tokens", type=int, default=64,
                        help="max new tokens for the Think stage")
    parser.add_argument("--print_generations", action="store_true", default=False,
                        help="print prompts, think, and response candidates")
    
    # 输出参数
    parser.add_argument("--log_file", type=str,
                        default="./logs/stage2_test.log",
                        help="all output log file path")
    
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def load_stage2_model(args, logger):
    """加载Stage 2训练好的模型 - 正确的分层加载"""
    logger.info("="*60)
    logger.info("🔄 Loading Qwen3 Stage 2 model...")
    logger.info("   Architecture: Base + Stage1(merged) + Stage2(LoRA)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 1. 加载分词器（从Stage 2模型目录，包含扩展词汇）
        logger.info("📝 Loading tokenizer from Stage 2 model...")
        tokenizer = AutoTokenizer.from_pretrained(args.stage2_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # 设置左侧padding用于生成任务
        tokenizer.padding_side = "left"
        logger.info(f"✅ Tokenizer loaded, vocab size: {len(tokenizer)}")
        
        # 2. 加载基础模型 (扩展词汇表)
        logger.info("🏗️ Loading base model with expanded vocab...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        logger.info("✅ Base model loaded")
        
        # 3. 加载并合并Stage 1 LoRA权重 (SID映射能力)
        logger.info("🔧 Loading and merging Stage 1 LoRA (SID mapping)...")
        stage1_model = PeftModel.from_pretrained(base_model, args.stage1_model_path)
        merged_model = stage1_model.merge_and_unload()
        logger.info("✅ Stage 1 LoRA merged")
        
        # 4. 加载Stage 2 LoRA权重 (推荐增强能力)
        logger.info("🎯 Loading Stage 2 LoRA (recommendation enhancement)...")
        final_model = PeftModel.from_pretrained(merged_model, args.stage2_model_path)
        logger.info("✅ Stage 2 LoRA loaded")
        
        # 5. 可选：合并Stage 2权重以提高推理速度
        logger.info("🔀 Merging Stage 2 weights for inference...")
        model = final_model.merge_and_unload()
        logger.info("✅ All weights merged - ready for inference")
        
        model.eval()
        logger.info("="*60)
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load Stage 2 model: {e}")
        logger.error("模型架构说明:")
        logger.error("  Stage 2模型 = Base模型 + Stage1(SID映射) + Stage2(推荐增强)")
        logger.error("可能的解决方案:")
        logger.error("1. 检查base_model_path是否正确")
        logger.error("2. 检查stage1_model_path是否正确") 
        logger.error("3. 检查stage2_model_path是否正确")
        logger.error("4. 确保Stage 2训练已完成")
        raise

def load_stage1_model(args, logger):
    """加载Stage 1模型用于对比"""
    logger.info("="*60)
    logger.info("🔄 Loading Qwen3 Stage 1 model for comparison...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 1. 加载分词器（从Stage 1 LoRA模型目录）
        logger.info("📝 Loading tokenizer from Stage 1 model...")
        tokenizer = AutoTokenizer.from_pretrained(args.stage1_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # 设置左侧padding用于生成任务
        tokenizer.padding_side = "left"
        logger.info(f"✅ Tokenizer loaded, vocab size: {len(tokenizer)}")
        
        # 2. 加载基础模型
        logger.info("🏗️ Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        logger.info("✅ Base model loaded")
        
        # 3. 加载并应用LoRA权重
        logger.info("🔧 Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, args.stage1_model_path)
        logger.info("✅ LoRA adapter loaded")
        
        # 4. 合并LoRA权重（提高推理速度）
        logger.info("🔀 Merging LoRA weights...")
        model = model.merge_and_unload()
        logger.info("✅ Weights merged")
        
        model.eval()
        logger.info("="*60)
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load Stage 1 model: {e}")
        logger.error("可能的解决方案:")
        logger.error("1. 检查Stage 1模型路径是否正确")
        logger.error("2. 检查基础模型路径是否正确")
        raise

def test_tokenization(tokenizer, logger):
    """测试SID token的tokenization是否正确"""
    logger.info("=== SID Token测试 ===")
    
    test_sid = "<|sid_begin|><s_a_156><s_b_218><s_c_251><s_d_244><|sid_end|>"
    
    # 检查特殊token是否被正确识别
    tokens = tokenizer.tokenize(test_sid)
    logger.info(f"SID tokenization: {tokens[:5]}...")  # 只显示前5个token
    
    # 检查token ID
    token_ids = tokenizer.encode(test_sid, add_special_tokens=False)
    logger.info(f"Token IDs (first 5): {token_ids[:5]}")
    
    # 检查是否有UNK token
    unk_id = tokenizer.unk_token_id
    if unk_id in token_ids:
        logger.warning("⚠️  警告: 发现未知token (UNK)，SID token可能未正确加载")
        return False
    else:
        logger.info("✅ SID token识别正常")
        return True

def setup_logging(log_file):
    """设置详细日志 - 输出到文件和控制台"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger('stage2_test')
    logger.setLevel(logging.DEBUG)
    
    # 清除已有handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # 文件handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # formatter
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def run_model_test(model, tokenizer, test_loader, all_items, prefix_allowed_tokens, 
                   args, logger, model_name):
    """运行单个模型的测试"""
    logger.info(f"🚀 Starting {model_name} evaluation...")
    
    metrics = args.metrics.split(",")
    metrics_results = {}
    total = 0
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader, desc=f"Testing {model_name}")):
            inputs_texts = batch["inputs"]
            targets = batch["targets"]
            bs = len(targets)
            
            # === Stage 1: Think (optional) ===
            think_texts = [""] * bs
            if args.enable_cot and args.think_max_tokens > 0:
                logger.info(f"🤔 Starting CoT Think stage for batch {step}...")
                think_inputs_texts = [f"{msg}\nThink:" for msg in inputs_texts]
                
                # 调试：打印第一个think input
                if step < 3:
                    logger.info(f"Think input example: {think_inputs_texts[0][:200]}...")
                
                enc_think = tokenizer(
                    think_inputs_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=tokenizer.model_max_length
                )
                enc_think = {k: v.to(model.device) for k, v in enc_think.items()}

                think_output = model.generate(
                    input_ids=enc_think["input_ids"],
                    attention_mask=enc_think.get("attention_mask", None),
                    max_new_tokens=args.think_max_tokens,
                    num_beams=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=False,
                    early_stopping=True,
                )
                think_decoded = tokenizer.batch_decode(think_output["sequences"], skip_special_tokens=True)
                
                # 调试：打印原始think输出
                if step < 3:
                    logger.info(f"Raw think output example: {think_decoded[0][:300]}...")

                # 提取每条的 Think 文本
                for i in range(bs):
                    full_text = think_decoded[i]
                    if "\nThink:" in full_text:
                        think_part = full_text.split("\nThink:")[-1]
                    else:
                        think_part = ""
                    if "Response:" in think_part:
                        think_part = think_part.split("Response:")[0]
                    think_texts[i] = think_part.strip()
                    
                    # 调试：输出提取的think文本
                    if step < 3:
                        logger.info(f"Extracted think text {i}: '{think_texts[i]}'")
                        
                logger.info(f"✅ CoT Think stage completed for batch {step}")
            elif args.enable_cot:
                logger.info("⚠️  CoT enabled but think_max_tokens=0, skipping Think stage")

            # === Stage 2: Response (constrained) ===
            if args.enable_cot and args.think_max_tokens > 0:
                response_inputs_texts = [f"{msg}\nThink:{think_texts[i]}\nResponse:" for i, msg in enumerate(inputs_texts)]
                # 调试：显示包含think的response input
                if step < 3:
                    logger.info(f"Response input with CoT: {response_inputs_texts[0][:300]}...")
            else:
                response_inputs_texts = [f"{msg}\nResponse:" for msg in inputs_texts]
            
            # 编码输入
            enc = tokenizer(
                response_inputs_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            
            # beam search生成
            num_beams = args.num_beams
            while True:
                try:
                    output = model.generate(
                        input_ids=enc["input_ids"],
                        attention_mask=enc.get("attention_mask", None),
                        max_new_tokens=10,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=num_beams,
                        num_return_sequences=num_beams,
                        output_scores=True,
                        return_dict_in_generate=True,
                        early_stopping=True,
                    )
                    break
                except RuntimeError as e:
                    err = str(e).lower()
                    if "out of memory" in err or "cuda" in err:
                        logger.warning(f"CUDA OOM with beam={num_beams}. Reducing beam size.")
                        num_beams -= 1
                        if num_beams < 1:
                            raise RuntimeError("Beam search OOM even with beam=1") from e
                        torch.cuda.empty_cache()
                    else:
                        raise
            
            # 解码输出
            output_ids = output["sequences"]
            scores = output.get("sequences_scores", None)
            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            # 详细日志记录和生成结果打印
            if args.print_generations:
                if scores is not None:
                    if hasattr(scores, 'detach'):
                        scores_list = [float(s) for s in scores.detach().cpu().tolist()]
                    else:
                        scores_list = [float(s) for s in scores]
                else:
                    scores_list = [float('nan')] * len(decoded)

                for i in range(bs):
                    start = i * num_beams
                    end = start + num_beams
                    cands = decoded[start:end]
                    cand_scores = scores_list[start:end]
                    
                    # 使用info级别确保控制台也能看到
                    logger.info(f"----- {model_name} SAMPLE {step*bs + i} -----")
                    logger.info(f"PROMPT:\n{inputs_texts[i]}")
                    if args.enable_cot and think_texts[i]:
                        logger.info(f"THINK:\n{think_texts[i]}")
                    logger.info("RESPONSE_CANDIDATES:")
                    for j, (c, sc) in enumerate(zip(cands, cand_scores)):
                        response = c.split("Response:")[-1].strip() if "Response:" in c else c
                        logger.info(f"  Rank {j+1}: score={sc:.4f} text={response}")
                    logger.info(f"TARGET: {targets[i]}")
                    
                    # 添加分隔线便于阅读
                    logger.info("-" * 50)
            
            # 计算topk结果
            topk_res = get_topk_results(
                decoded, scores, targets, num_beams,
                all_items=all_items if args.filter_items else None
            )
            
            # 累积metrics
            batch_metrics_res = get_metrics_results(topk_res, metrics)
            for m, res in batch_metrics_res.items():
                metrics_results[m] = metrics_results.get(m, 0.0) + res
            total += bs
            
            # 中间进度汇报
            if (step + 1) % 50 == 0:
                temp_results = {m: metrics_results[m] / total for m in metrics_results}
                logger.info(f"[{model_name} Progress] Step {step+1}, Metrics: {temp_results}")
    
    # 计算最终结果
    for m in metrics_results:
        metrics_results[m] = metrics_results[m] / total if total > 0 else 0.0
    
    return metrics_results

def run_stage2_test(args, logger=None):
    """运行Stage 2测试 - 使用训练时预留的验证数据"""
    set_seed(args.seed)
    
    # 如果没有传入logger，创建一个
    if logger is None:
        logger = setup_logging(args.log_file)
    
    logger.info("🧪 Starting Qwen3 Stage 2 Hit Rate Test")
    logger.info(f"Args: {vars(args)}")
    
    # 加载Stage 2验证数据集（训练时预留的数据）
    logger.info("📊 Loading Stage 2 validation dataset...")
    logger.info(f"   Data source: {args.stage2_val_data_path}")
    logger.info("   Using Stage 2 validation data (preserved from training)")
    
    test_data = Stage2ValDataset(args.stage2_val_data_path, sample_num=args.sample_num)
    all_items = test_data.get_all_items()
    logger.info(f"📈 Test data size: {len(test_data)}")
    
    # 测试完整的Stage 2模型 (Base + Stage1 + Stage2)
    logger.info("\n" + "="*80)
    logger.info("🎯 Testing Complete Stage 2 Model")
    logger.info("    Architecture: Base + Stage1(SID映射) + Stage2(推荐增强)")
    logger.info("="*80)
    
    stage2_model, stage2_tokenizer = load_stage2_model(args, logger)
    
    # 测试tokenization
    if not test_tokenization(stage2_tokenizer, logger):
        logger.warning("Stage 2 SID tokenization可能有问题，但继续测试...")
    
    stage2_collator = TestCollator(args, stage2_tokenizer)
    stage2_prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(stage2_tokenizer)
    
    stage2_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=stage2_collator,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    results = run_model_test(
        stage2_model, stage2_tokenizer, stage2_loader, all_items, 
        stage2_prefix_allowed_tokens, args, logger, "Complete Stage 2"
    )
    
    # 输出最终结果
    logger.info("\n" + "="*80)
    logger.info("📊 FINAL RESULTS")
    logger.info("="*80)
    
    logger.info("🎯 Complete Stage 2 Model Results:")
    for metric, value in results.items():
        logger.info(f"  {metric:>10}: {value:.4f}")
    
    logger.info("="*80)
    
    # 输出测试摘要
    logger.info("\n📋 Test Summary:")
    logger.info(f"Base Model: {args.base_model_path}")
    logger.info(f"Stage 1 Model: {args.stage1_model_path}")
    logger.info(f"Stage 2 Model: {args.stage2_model_path}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Total samples: {len(test_data)}")
    logger.info(f"Batch size: {args.test_batch_size}")
    logger.info(f"Beam size: {args.num_beams}")
    logger.info(f"CoT enabled: {args.enable_cot}")
    if args.enable_cot:
        logger.info(f"Think max tokens: {args.think_max_tokens}")
    
    logger.info("\n✅ Complete Stage 2 model testing completed successfully!")
    
    return results

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志（先创建以便记录错误）
    logger = setup_logging(args.log_file)
    
    try:
        results = run_stage2_test(args, logger)
        logger.info("✅ Stage 2 testing completed successfully!")
        return True
    except Exception as e:
        # 将错误同时输出到日志和控制台
        import traceback
        error_msg = f"❌ Testing failed: {e}"
        logger.error(error_msg)
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)