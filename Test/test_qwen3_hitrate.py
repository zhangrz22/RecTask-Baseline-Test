#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 SID映射模型的Hit Rate测试脚本
基于原有test_hitrate.py，专门适配Qwen3训练好的模型
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

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 Hit Rate Test")
    
    # 基础参数
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_path", type=str, default="./", help="data directory")
    parser.add_argument("--dataset", type=str, default="Beauty", help="Dataset name")
    parser.add_argument("--index_file", type=str, default=".index.json", help="the item indices file")
    
    # 模型路径参数 - 针对Qwen3优化
    parser.add_argument("--base_model_path", type=str, 
                        default="../Qwen3/model/Qwen3-1-7B-expanded-vocab",
                        help="Base model path (expanded vocab)")
    parser.add_argument("--lora_model_path", type=str,
                        default="../Qwen3/results/sid_mapping_model", 
                        help="LoRA model path")
    
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
                        default="./logs/qwen3_test.log",
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

def load_qwen3_model(args, logger):
    """加载Qwen3训练好的模型"""
    logger.info("="*60)
    logger.info("🔄 Loading Qwen3 model with PEFT...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 1. 加载分词器（从LoRA模型目录，包含SID tokens）
        logger.info("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.lora_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"✅ Tokenizer loaded, vocab size: {len(tokenizer)}")
        
        # 2. 加载基础模型
        logger.info("🏗️ Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        logger.info("✅ Base model loaded")
        
        # 3. 加载并应用LoRA权重
        logger.info("🔧 Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, args.lora_model_path)
        logger.info("✅ LoRA adapter loaded")
        
        # 4. 合并LoRA权重（提高推理速度）
        logger.info("🔀 Merging LoRA weights...")
        model = model.merge_and_unload()
        logger.info("✅ Weights merged")
        
        model.eval()
        logger.info("="*60)
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error("可能的解决方案:")
        logger.error("1. 检查模型路径是否正确")
        logger.error("2. 确保PEFT库已安装: pip install peft") 
        logger.error("3. 检查基础模型是否存在")
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
    """设置详细日志 - 只输出到文件"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger('qwen3_test')
    logger.setLevel(logging.DEBUG)
    
    # 清除已有handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # 只使用文件handler，不输出到控制台
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # formatter
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger

def run_hitrate_test(args):
    """运行Hit Rate测试"""
    set_seed(args.seed)
    
    # 设置日志
    logger = setup_logging(args.log_file)
    logger.info("🧪 Starting Qwen3 Hit Rate Test")
    logger.info(f"Args: {vars(args)}")
    
    # 1. 加载模型
    model, tokenizer = load_qwen3_model(args, logger)
    
    # 2. 测试tokenization
    if not test_tokenization(tokenizer, logger):
        logger.warning("SID tokenization可能有问题，但继续测试...")
    
    # 3. 加载数据集
    logger.info("📊 Loading test dataset...")
    test_data = SeqRecDataset(args, "test", sample_num=args.sample_num)
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()
    prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)
    
    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    logger.info(f"📈 Test data size: {len(test_data)}")
    
    # 4. 开始测试
    metrics = args.metrics.split(",")
    metrics_results = {}
    total = 0
    
    logger.info("🚀 Starting Hit Rate evaluation...")
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader, desc="Testing")):
            inputs_texts = batch["inputs"]
            targets = batch["targets"]
            bs = len(targets)
            
            # === Stage 1: Think (optional) ===
            think_texts = [""] * bs
            if args.enable_cot:
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

            # === Stage 2: Response (constrained) ===
            if args.enable_cot:
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
                    logger.info(f"----- SAMPLE {step*bs + i} -----")
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
                logger.info(f"[Progress] Step {step+1}, Metrics: {temp_results}")
    
    # 5. 最终结果
    for m in metrics_results:
        metrics_results[m] = metrics_results[m] / total if total > 0 else 0.0
    
    logger.info("="*60)
    logger.info("🎯 Final Hit Rate Results:")
    logger.info("="*60)
    for metric, value in metrics_results.items():
        logger.info(f"{metric:>10}: {value:.4f}")
    logger.info("="*60)
    
    # 6. 输出结果摘要到日志
    logger.info("\n📊 Test Summary:")
    logger.info(f"Model: {args.base_model_path} + {args.lora_model_path}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Total samples: {total}")
    logger.info(f"Batch size: {args.test_batch_size}")
    logger.info(f"Beam size: {args.num_beams}")
    logger.info(f"CoT enabled: {args.enable_cot}")
    if args.enable_cot:
        logger.info(f"Think max tokens: {args.think_max_tokens}")
    
    logger.info("\n✅ Testing completed successfully!")
    
    return metrics_results

def main():
    """主函数"""
    args = parse_args()
    
    try:
        results = run_hitrate_test(args)
        return True
    except Exception as e:
        # 只在出错时输出到控制台
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)