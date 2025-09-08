

sft_prompt = "You are an expert in Recommender System. The user has interacted with several items in chronological order. Can you predict the next possible item that the user may expect?" \
             "\n\n### {instruction}\n\n### {response}"

# 简化：直接定义单个prompt
prompt = {
    "instruction": "The user's previous interaction history is as follows: {inters}",
    "response": "{item}"
}

