import json
from tqdm import tqdm
from collections import deque

import torch
import datasets
from vllm import LLM, SamplingParams
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


model_path = 'Qwen/Qwen2.5-Math-7B-Instruct'

model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation='flash_attention_2', torch_dtype='bfloat16').to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_path)

idx = 4
bsz = 16
data = json.load(open('1.5B_1.json'))
response = data[idx]['response'].replace('<think>', '').strip()
steps = response.split('\n\n')
problem = data[idx]['problem']
choices = deque(['8', '9', '10', '12'])
gt_idx = 1
results = []
system_prompt = "Based on the provided partial reasoning, answer the multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D)."
user_prompt = "Question:\n{question}\n\nThoughts:\n{cot}\n\nOptions:\n{options}"
assistant_prompt = "Thinking" + '...'*50 + '\nThe answer is'
# assistant_prompt = 'The answer is'

choice_idxs = [362, 425, 356, 422]

import pdb; pdb.set_trace()
for i in range(0, 32, bsz):
    messages = []
    for j in range(i, i+bsz):
        for k in range(4):
            options = f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
            messages.append([{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt.format(question=problem, cot='\n\n'.join(steps[:j]), options=options)}, {'role': 'assistant', 'content': assistant_prompt}])
            choices.rotate(1)
    templated_convs = [tokenizer.apply_chat_template(m, continue_final_message=True, tokenize=False) for m in messages]
    inputs = tokenizer(templated_convs, return_tensors="pt", padding=True, add_special_tokens=False).to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        last_token_logits = logits[torch.arange(logits.shape[0]), inputs['attention_mask'].sum(dim=-1)-1]
        probabilities = torch.softmax(last_token_logits, dim=-1)
        probabilities = probabilities[:, [362, 425, 356, 422]]
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        gt_idxs = [gt_idx, (gt_idx+1)%4, (gt_idx+2)%4, (gt_idx+3)%4] * (len(templated_convs) // 4)
        gt_probs = probabilities[torch.arange(probabilities.shape[0]), gt_idxs]
        gt_probs = gt_probs.reshape(-1, 4)
        gt_probs = gt_probs.mean(dim=-1)
    results.append({'problem': item['problem'], 'response': outs, 'answer': item['answer']})

json.dump(results, open('1.5B_1.json', 'w'), indent=2)
