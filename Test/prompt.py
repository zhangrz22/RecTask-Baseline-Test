

sft_prompt = "You are an expert in Recommender System. The user has interacted with several items in chronological order. Can you predict the next possible item that the user may expect?" \
             "\n\n### {instruction}\n\n### {response}"

all_prompt = {}

# =====================================================
# Task 1 -- Sequential Recommendation -- 17 Prompt
# =====================================================

seqrec_prompt = []

#####——0
prompt = {}
prompt["instruction"] = "The user's previous interaction history is as follows: {inters}"
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)


all_prompt["seqrec"] = seqrec_prompt

