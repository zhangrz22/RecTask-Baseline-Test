#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试SID映射模型的推理效果
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

# --- 配置区域 ---
# 基础模型路径（扩展词汇后的模型）
BASE_MODEL_PATH = "/home/liuzhanyu/Rec_baseline/RecTask-Baseline-Test/Qwen3/model/Qwen3-1-7B-expanded-vocab"
# LoRA模型路径
LORA_MODEL_PATH = "/home/liuzhanyu/Rec_baseline/RecTask-Baseline-Test/Qwen3/results/sid_mapping_model"

# 测试用例：匹配新的训练格式（包含think标签）
TEST_CASES = [
    # 问答格式（与训练时一致）
    "What is <|sid_begin|><s_a_156><s_b_218><s_c_251><s_d_244><|sid_end|>?",
    "Can you tell me what <|sid_begin|><s_a_156><s_b_218><s_c_251><s_d_244><|sid_end|> represents?",
    "Q: What does <|sid_begin|><s_a_156><s_b_218><s_c_251><s_d_244><|sid_end|> represent? A:",
    "Identify this product: <|sid_begin|><s_a_156><s_b_218><s_c_251><s_d_244><|sid_end|>",
]

# 从实际训练数据中获取的真实SID-标题对
GROUND_TRUTH_CASES = [
    {
        "sid": "<|sid_begin|><s_a_156><s_b_218><s_c_251><s_d_244><|sid_end|>",
        "expected_title": "MoYo Natural Labs 4 oz Travel Bottles, Empty Travel Containers with Flip Caps, BPA Free PET Plastic Squeezable Toiletry/Cosmetic Bottles (Neck 20-410) (Pack of 30, Clear)",
        "prompt": "<|sid_begin|><s_a_156><s_b_218><s_c_251><s_d_244><|sid_end|> represents"
    }
]

def test_tokenization(tokenizer):
    """测试SID token的tokenization是否正确"""
    print("\n=== Token化测试 ===")
    
    test_sid = "<|sid_begin|><s_a_156><s_b_218><s_c_251><s_d_244><|sid_end|>"
    
    # 检查特殊token是否被正确识别
    tokens = tokenizer.tokenize(test_sid)
    print(f"原始SID: {test_sid}")
    print(f"Token化结果: {tokens}")
    
    # 检查token ID
    token_ids = tokenizer.encode(test_sid, add_special_tokens=False)
    print(f"Token IDs: {token_ids}")
    
    # 反向解码
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    print(f"反向解码: {decoded}")
    
    # 检查是否有UNK token
    unk_id = tokenizer.unk_token_id
    if unk_id in token_ids:
        print("⚠️  警告: 发现未知token (UNK)，SID token可能未正确加载")
    else:
        print("✅ SID token识别正常")
    
    return len([t for t in tokens if '<s_' in t or '<|sid_' in t])

def load_model_and_tokenizer():
    """正确加载基础模型和LoRA权重"""
    print("="*60)
    print("🔄 开始加载模型...")
    start_time = time.time()
    
    try:
        # 1. 加载分词器（从LoRA模型目录，包含SID tokens）
        print("📝 加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ 分词器加载成功，词汇表大小: {len(tokenizer)}")
        
        # 2. 加载基础模型
        print("🏗️  加载基础模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print("✅ 基础模型加载成功")
        
        # 3. 加载并应用LoRA权重
        print("🔧 加载LoRA适配器...")
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
        print("✅ LoRA适配器加载成功")
        
        # 4. 合并LoRA权重（可选，提高推理速度）
        print("🔀 合并LoRA权重...")
        model = model.merge_and_unload()
        print("✅ 权重合并完成")
        
        load_time = time.time() - start_time
        print(f"⏱️  总加载时间: {load_time:.2f}秒")
        print("="*60)
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查模型路径是否正确")
        print("2. 安装PEFT库: pip install peft")
        print("3. 确保基础模型存在")
        raise

def test_inference(model, tokenizer, prompt, max_new_tokens=50):
    """执行单次推理测试"""
    print(f"\n📝 测试prompt: {prompt}")
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,   # 使用采样避免重复
            temperature=0.3,  # 降低温度减少随机性
            top_p=0.8,        # 使用nucleus采样
            repetition_penalty=1.2,  # 重复惩罚
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,  # 遇到EOS就停止
        )
    
    # 解码输出（只获取新生成的部分）
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    print(f"🤖 模型回答: {response}")
    return response

def test_ground_truth(model, tokenizer):
    """测试训练数据中的真实映射"""
    print("\n" + "="*60)
    print("🎯 Ground Truth测试（训练数据中的真实映射）")
    
    for i, case in enumerate(GROUND_TRUTH_CASES, 1):
        print(f"\n--- Ground Truth测试 {i} ---")
        print(f"SID: {case['sid']}")
        print(f"预期标题: {case['expected_title'][:80]}...")
        
        response = test_inference(model, tokenizer, case['prompt'], max_new_tokens=80)
        
        # 检查关键词匹配
        expected_keywords = ['moyo', 'natural', 'labs', 'travel', 'bottles']
        found_keywords = [kw for kw in expected_keywords if kw in response.lower()]
        
        print(f"找到关键词: {found_keywords}")
        
        if len(found_keywords) >= 3:
            print("✅ 匹配良好！")
        elif len(found_keywords) >= 1:
            print("⚠️  部分匹配")
        else:
            print("❌ 匹配失败")

def main():
    """主测试函数"""
    try:
        # 1. 加载模型
        model, tokenizer = load_model_and_tokenizer()
        
        # 2. 测试tokenization
        num_sid_tokens = test_tokenization(tokenizer)
        print(f"✅ 识别到 {num_sid_tokens} 个SID相关token")
        
        # 3. Ground Truth测试（优先）
        test_ground_truth(model, tokenizer)
        
        # 4. 通用推理测试
        print("\n" + "="*60)
        print("🧪 通用推理测试...")
        
        for i, test_case in enumerate(TEST_CASES, 1):
            print(f"\n--- 测试用例 {i} ---")
            response = test_inference(model, tokenizer, test_case)
            
            # 简单检查是否包含商品相关信息
            if any(keyword in response.lower() for keyword in ['travel', 'bottle', 'moyo', 'natural', 'labs']):
                print("✅ 回答似乎包含相关商品信息")
            else:
                print("⚠️  回答可能不够准确")
        
        print("\n" + "="*60)
        print("🎉 测试完成！")
        
        # 5. 总结和建议
        print("\n💡 如果结果不理想，可能的原因:")
        print("1. 训练不充分 - 需要更多epochs或更大学习率")
        print("2. 数据质量问题 - 检查训练数据格式")
        print("3. 生成参数设置 - 调整temperature和repetition_penalty")
        print("4. 模型容量问题 - 可能需要训练更多参数")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()