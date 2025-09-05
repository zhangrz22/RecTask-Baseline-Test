import torch
import copy
import argparse
from dataclasses import dataclass

import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration


class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        full_texts = [d["labels"] + self.tokenizer.eos_token for d in batch]

        inputs = self.tokenizer(
            text = full_texts,
            text_target = input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        labels = copy.deepcopy(inputs["input_ids"])
        if self.only_train_response:
            # ignore padding
            labels[labels == self.tokenizer.pad_token_id] = -100
            # ignore input text
            labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100

        inputs["labels"] = labels


        return inputs



class TestCollator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        if isinstance(self.tokenizer, LlamaTokenizer):
            self.tokenizer.padding_side = "left"

    def __call__(self, batch):
        # 构造 batch_prompts，同时保留原始 labels
        batch_prompts = []
        targets = [d["labels"] for d in batch]
        print(f"batch is {batch}")

        for d in batch:
            message = d["input_ids"]   # 原始文本
            v = d.get("v", None)       # v 可能在 batch 中

            system_text = "You are an expert in Recommender System. The user has interacted with several items in chronological order. Can you predict the next possible item that the user may expect?"
            if system_text in message:
                user_text = message.split(system_text)[-1].strip()
                chat_format = [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text}
                ]
            else:
                # fallback: 如果找不到 system_text，则全当作 user
                chat_format = [
                    {"role": "user", "content": message}
                ]
            print(f"chat_format == {chat_format}")

            prompt_text = self.tokenizer.apply_chat_template(
                chat_format,
                tokenize=False,
                enable_thinking=True,
                add_generation_prompt=True
            )

            batch_prompts.append(prompt_text)

        print(f"batch_prompts == {batch_prompts}")


        # 返回 tokenized inputs + targets + 原始 prompt_text（可选，用于 llm.generate）
        return {
            "inputs": batch_prompts,
            "targets": targets
        }

