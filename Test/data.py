import copy
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import pdb
import json
from prompt import sft_prompt, prompt
import numpy as np


class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.add_prefix = args.add_prefix

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None


    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_prefix_allowed_tokens_fn(self, tokenizer):

        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
                    if not token_ids:
                        # 跳过无法分词成 token 的条目
                        continue
                    token_id = token_ids[0]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            # 第 len(self.allowed_tokens) 位只允许结束
            eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([eos_id])

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
            # 在已生成前缀中定位 "Response:" 的最后一次出现
            pos = find_last_sublist(sentence, sep)
            if pos is None:
                # 未定位到分隔符时，不进行约束
                # 返回全词表以避免报错（HF 期望返回可用 token 列表）
                try:
                    vocab_size = getattr(tokenizer, 'vocab_size', None) or len(tokenizer)
                except Exception:
                    vocab_size = 50257
                return list(range(vocab_size))

            # 第几个 response token（从 0 开始）
            gen_pos = len(sentence) - (pos + len(sep))
            if gen_pos in self.allowed_tokens:
                return list(self.allowed_tokens[gen_pos])
            else:
                # 超过最大长度后，只允许结束
                last_key = max(self.allowed_tokens.keys())
                return list(self.allowed_tokens[last_key])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError



class SeqRecDataset(BaseDataset):
        
    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        # 简化：直接使用固定的prompt
        self.prompt = prompt


        # load data
        self._load_data()
        self._remap_items()
        
        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.sample_valid = args.sample_valid
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)


    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items


    def _process_train_data(self):

        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = self.his_sep.join(history)
                inter_data.append(one_data)

        return inter_data
    
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.inter_data) * self.prompt_sample_num
        elif self.mode == 'valid':
            return len(self.valid_text_data)
        elif self.mode == 'test':
            return len(self.inter_data)
        else:
            raise NotImplementedError
                    
    def _construct_valid_text(self):
        self.valid_text_data = []
        # 简化：直接使用固定prompt
        for i in range(len(self.inter_data)):
            d = self.inter_data[i]
            input, output = self._get_text_data(d, self.prompt)
            self.valid_text_data.append({"input_ids": input, "labels": output})

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        if self.mode == 'test':
            # print(f"input = {input}, output = {response}")
            return input, response

        return input, output

    def __getitem__(self, index):

        if self.mode == 'valid':
            return self.valid_text_data[index]

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]
        # print(index, idx)

        # 简化：直接使用固定prompt
        input, output = self._get_text_data(d, self.prompt)

        print({"input": input, "output": output})

        return dict(input_ids=input, labels=output)

