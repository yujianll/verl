# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets
import json

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from verl.utils.reward_score.qwen_math import extract_answer


def extract_solution(solution_str):
    # return remove_boxed(last_boxed_only_string(solution_str))
    return extract_answer(solution_str, 'math')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_file = '/data/yujian_liu/math/data/math_level3to5_data_processed_with_qwen_prompt.json'
    train_dataset = datasets.load_dataset('json', data_files=[data_file])['train']
    train_dataset = train_dataset.remove_columns("answer").rename_column("gt_answer", "answer").rename_column("question", "problem")

    test_dataset = datasets.load_dataset("HuggingFaceH4/MATH-500")['test']
    
    data_source = 'math'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')
            answer = example.pop('answer')
            solution = extract_solution(answer)
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    keep_columns = {'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'}
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=list(set(train_dataset.features) - keep_columns))
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=list(set(test_dataset.features) - keep_columns))

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
