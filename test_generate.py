import json
from tqdm import tqdm

import datasets
from vllm import LLM, SamplingParams
from transformers import pipeline, AutoTokenizer


model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
# llm = LLM(
#     model=model_path,
#     gpu_memory_utilization=0.9,
#     enable_prefix_caching=True,
#     seed=0,
#     tensor_parallel_size=2
# )
# sampling_params = SamplingParams(
#     temperature=0.8,
#     max_tokens=4096,
#     top_p=1.0,
#     seed=0
# )
# tokenizer = llm.get_tokenizer()

pipe = pipeline("text-generation", model=model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


math500 = datasets.load_dataset("HuggingFaceH4/MATH-500")['test']
math500 = math500.filter(lambda x: x['level'] == 5).select(range(25, 35))
results = []

# import pdb; pdb.set_trace()
for item in tqdm(math500):
    messages = [{'role': 'user', 'content': item['problem']}]
    # templated_convs = tokenizer.apply_chat_template(
    #     messages,
    #     add_generation_prompt=True,
    #     tokenize=False,
    # )
    # llm_outputs = llm.generate(templated_convs, sampling_params, use_tqdm=False)[0].outputs[0].text
    outs = pipe(messages, do_sample=True, temperature=0.8, max_new_tokens=4096, stop_sequence=tokenizer.eos_token)[0]['generated_text'][-1]['content']
    results.append({'problem': item['problem'], 'response': outs, 'answer': item['answer']})

json.dump(results, open('1.5B_1.json', 'w'), indent=2)
