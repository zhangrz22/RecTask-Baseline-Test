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
from prompt import all_prompt
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



os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

def parse_global_args(parser):
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--base_model", type=str,
                        default="./llama-7b/",
                        help="basic model path")
    parser.add_argument("--output_dir", type=str,
                        default="./ckpt/",
                        help="The output directory")

    return parser

def parse_dataset_args(parser):
    parser.add_argument("--data_path", type=str, default="./data/",
                        help="data directory")
    # 已简化：固定为seqrec任务
    parser.add_argument("--dataset", type=str, default="Beauty", help="Dataset name")
    parser.add_argument("--index_file", type=str, default=".index.json", help="the item indices file")

    # arguments related to sequential task
    parser.add_argument("--max_his_len", type=int, default=20,
                        help="the max number of items in history sequence, -1 means no limit")
    parser.add_argument("--add_prefix", action="store_true", default=False,
                        help="whether add sequential prefix in history")
    parser.add_argument("--his_sep", type=str, default=", ", help="The separator used for history")
    parser.add_argument("--only_train_response", action="store_true", default=False,
                        help="whether only train on responses")

    # 已简化：不再需要多任务和多prompt相关参数

    return parser

def parse_test_args(parser):

    parser.add_argument("--ckpt_path", type=str,
                        default="",
                        help="The checkpoint path")
    parser.add_argument("--lora", action="store_true", default=True)
    parser.add_argument("--filter_items", action="store_true", default=False,
                        help="whether filter illegal items")

    parser.add_argument("--results_file", type=str,
                        default="./results/test-ddp.json",
                        help="result output path")

    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=-1,
                        help="test sample number, -1 represents using all test data")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID when testing with single GPU")
    # 已简化：不再需要prompt选择参数
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
                        help="test metrics, separate by comma")
    # 已简化：固定为seqrec任务

    # CoT/diagnostics
    parser.add_argument("--enable_cot", action="store_true", default=False,
                        help="enable two-stage generation: Think then constrained Response")
    parser.add_argument("--think_max_tokens", type=int, default=64,
                        help="max new tokens for the Think stage")
    parser.add_argument("--print_generations", action="store_true", default=False,
                        help="print prompts, think, and response candidates")


    return parser


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def load_test_dataset(args):
    return SeqRecDataset(args, "test", sample_num=args.sample_num)


def test_ddp(args):
    set_seed(args.seed)

    # 环境变量通常由 torchrun 设置
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    rank = int(os.environ.get("RANK", local_rank))

    # 设置当前进程的 GPU
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 初始化进程组
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    if local_rank == 0:
        print("args:", vars(args))
    # 加载 tokenizer 与模型（不要使用 device_map="auto"）
    model_path = args.ckpt_path if args.ckpt_path else "/llm-reco-ssd-share/baohonghui/think_pretrain/results/pretrain_only_te/hf_model_step3234_final"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.to(device)
    model.eval()

    # 包装为 DDP
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )

    # --- 数据与 sampler ---
    if args.test_prompt_ids == "all":
        if args.test_task.lower() == "seqrec":
            prompt_ids = range(len(all_prompt["seqrec"]))
        elif args.test_task.lower() == "itemsearch":
            prompt_ids = range(len(all_prompt["itemsearch"]))
        elif args.test_task.lower() == "fusionseqrec":
            prompt_ids = range(len(all_prompt["fusionseqrec"]))
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    test_data = load_test_dataset(args)
    ddp_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, drop_last=False)
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()
    prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)
    # print(f"prefix_allowed_tokens = {prefix_allowed_tokens}")
    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        sampler=ddp_sampler,
        num_workers=2,
        pin_memory=True
    )

    if local_rank == 0:
        print("data num:", len(test_data))

    metrics = args.metrics.split(",")
    all_prompt_results = []

    with torch.no_grad():
        for prompt_id in prompt_ids:
            # set epoch -> 保证 DistributedSampler 每个进程取到不同 subset（关键）
            ddp_sampler.set_epoch(prompt_id)

            if local_rank == 0:
                print("Start prompt:", prompt_id)

            # 如果你的 dataset 支持按 prompt 切换（如 set_prompt）
            test_loader.dataset.set_prompt(prompt_id)

            metrics_results = {}
            total = 0

            for step, batch in enumerate(tqdm(test_loader, desc=f"Rank{rank} Prompt{prompt_id}")):
                # collator 返回 {"inputs": list[str], "targets": list[str]}
                inputs_texts = batch["inputs"]
                targets = batch["targets"]

                enc = tokenizer(
                    inputs_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=tokenizer.model_max_length
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                bs = len(targets)

                num_beams = args.num_beams
                while True:
                    try:
                        # 使用 model.module.generate，因为 model 被 DDP 包裹
                        output = model.module.generate(
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
                        # 在 HF + CUDA 下，OOM 常表现为 RuntimeError containing "out of memory"
                        err = str(e).lower()
                        if "out of memory" in err or "cuda" in err:
                            if local_rank == 0:
                                print(f"[rank {rank}] CUDA OOM with beam={num_beams}. Reducing beam.")
                            num_beams -= 1
                            if num_beams < 1:
                                raise RuntimeError("Beam search OOM even with beam=1") from e
                            # 清理显存并重试
                            torch.cuda.empty_cache()
                        else:
                            # 其它异常直接抛出（便于调试）
                            raise

                # HF generate 返回 sequences 和 sequences_scores（如果 return_dict_in_generate=True）
                output_ids = output["sequences"]
                scores = output.get("sequences_scores", None)

                decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)


                topk_res = get_topk_results(decoded, scores, targets, num_beams,
                                            all_items=all_items if args.filter_items else None)

                # --- 汇总 batch size（标量） ---
                bs_tensor = torch.tensor(bs, device=device, dtype=torch.long)
                dist.all_reduce(bs_tensor, op=dist.ReduceOp.SUM)
                total += int(bs_tensor.item())

                # --- 汇总 topk_res（list），需要 all_gather_object ---
                res_gather_list = [None for _ in range(world_size)]
                dist.all_gather_object(object_list=res_gather_list, obj=topk_res)

                # 只有主进程计算 metrics 并打印
                if local_rank == 0:
                    all_device_topk_res = []
                    for ga_res in res_gather_list:
                        if ga_res:
                            all_device_topk_res += ga_res

                    batch_metrics_res = get_metrics_results(all_device_topk_res, metrics)
                    for m, res in batch_metrics_res.items():
                        metrics_results[m] = metrics_results.get(m, 0.0) + res

                    if (step + 1) % 50 == 0:
                        temp = {m: metrics_results[m] / total for m in metrics_results}
                        print(f"[prompt {prompt_id} progress] averaged metrics:", temp)

                # 不需要每个 batch barrier，会降低性能
                # dist.barrier()

            # prompt 完成后同步并记录结果（主进程负责打印和保存）
            dist.barrier()
            if local_rank == 0:
                # 平均化 metrics_results
                for m in metrics_results:
                    metrics_results[m] = metrics_results[m] / total if total > 0 else 0.0

                all_prompt_results.append(metrics_results)
                print("======================================================")
                print("Prompt {} results: ".format(prompt_id), metrics_results)
                print("=========================")

def test_single(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 加载 tokenizer 和模型 ---
    model_path = args.ckpt_path if args.ckpt_path else "/llm-reco-ssd-share/baohonghui/think_pretrain/results/pretrain_only_te/hf_model_step3234_final"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.to(device)
    model.eval()


    # --- 数据 ---
    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()
    prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        shuffle=False,  # 单卡就不需要 DistributedSampler
        num_workers=2,
        pin_memory=True
    )

    print("data num:", len(test_data))

    metrics = args.metrics.split(",")
    metrics_results = {}
    total = 0

    with torch.no_grad():
        print("Starting evaluation...")
        
        for step, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                # TestCollator 返回 {"inputs": list[str], "targets": list[str]}
                inputs_texts = batch["inputs"]
                targets = batch["targets"]
                bs = len(targets)

                # === Stage 1: Think (optional) ===
                think_texts = [""] * bs
                if args.enable_cot:
                    think_inputs_texts = [f"{msg}\nThink:" for msg in inputs_texts]
                    enc_think = tokenizer(
                        think_inputs_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=tokenizer.model_max_length
                    )
                    enc_think = {k: v.to(device) for k, v in enc_think.items()}

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

                    # 提取每条的 Think 文本（以最后一次 "\nThink:" 为切分点），并去除后续的 "Response:" 片段
                    for i in range(bs):
                        full_text = think_decoded[i]
                        if "\nThink:" in full_text:
                            think_part = full_text.split("\nThink:")[-1]
                        else:
                            think_part = ""
                        if "Response:" in think_part:
                            think_part = think_part.split("Response:")[0]
                        think_texts[i] = think_part.strip()

                # === Stage 2: Response (constrained) ===
                if args.enable_cot:
                    response_inputs_texts = [f"{msg}\nThink:{think_texts[i]}\nResponse:" for i, msg in enumerate(inputs_texts)]
                else:
                    response_inputs_texts = [f"{msg}\nResponse:" for msg in inputs_texts]

                enc = tokenizer(
                    response_inputs_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=tokenizer.model_max_length
                )
                enc = {k: v.to(device) for k, v in enc.items()}

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
                            print(f"[OOM] CUDA OOM with beam={num_beams}. Reducing beam.")
                            num_beams -= 1
                            if num_beams < 1:
                                raise RuntimeError("Beam search OOM even with beam=1") from e
                            torch.cuda.empty_cache()
                        else:
                            raise

                output_ids = output["sequences"]
                scores = output.get("sequences_scores", None)
                decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                # 可选：打印生成（每条样本的 Prompt/Think/Top-k 候选与分数）
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
                        print("----- SAMPLE {} -----".format(i), flush=True)
                        print("PROMPT:\n{}".format(inputs_texts[i]), flush=True)
                        if args.enable_cot:
                            print("THINK:\n{}".format(think_texts[i]), flush=True)
                        print("RESPONSE_CANDIDATES:", flush=True)
                        for c, sc in zip(cands, cand_scores):
                            try:
                                print(f"  - score={sc:.4f} text={c}", flush=True)
                            except Exception:
                                print(f"  - score={sc} text={c}", flush=True)

                topk_res = get_topk_results(
                    decoded, scores, targets, num_beams,
                    all_items=all_items if args.filter_items else None
                )

                # 累积 metrics
                batch_metrics_res = get_metrics_results(topk_res, metrics)
                for m, res in batch_metrics_res.items():
                    metrics_results[m] = metrics_results.get(m, 0.0) + res
                total += bs

                if (step + 1) % 50 == 0:
                    temp = {m: metrics_results[m] / total for m in metrics_results}
                    print(f"[progress] averaged metrics:", temp)

        # 计算最终平均指标
        for m in metrics_results:
            metrics_results[m] = metrics_results[m] / total if total > 0 else 0.0

        print("======================================================")
        print("Final results:", metrics_results)
        print("=========================")

    return metrics_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()
    # test_ddp(args)
    test_single(args)
