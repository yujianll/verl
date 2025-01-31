# This file defines a simple interface for evaluation

import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from functools import partial

from verl.utils.registry_utils import Registry
from verl.utils.reward_score.qwen_math import math_equal as QwenMathEqual
from verl.utils.reward_score.qwen_math import extract_answer as QwenExtractAnswer
from verl.utils.reward_score.deepseek_math import math_equal as DeepSeekMathEqual
from verl.utils.reward_score.deepseek_math import (
    extract_answer as DeepSeekExtractAnswer,
)
from verl.utils.reward_score.math import compute_score as VerLMathEqual


REWARD_REGISTRY = Registry()


def qwen_math_equal(prediction, gt_answer):
    pred = QwenExtractAnswer(prediction, "math")
    return QwenMathEqual(pred, gt_answer, timeout=True)


def deepseek_math_equal(prediction, gt_answer):
    pred = DeepSeekExtractAnswer(prediction)
    return DeepSeekMathEqual(pred, gt_answer, timeout=True)


REWARD_REGISTRY.register("qwen_math", qwen_math_equal)
REWARD_REGISTRY.register("deepseek_math", deepseek_math_equal)
REWARD_REGISTRY.register("verl_math", VerLMathEqual)


def pass_at_k(results: List[Dict[str, Any]], k) -> float:
    acc_mapping = defaultdict(list)
    for result in results:
        acc_mapping[result["uid"]].append(result["match"])
    return pass_at_k_mapping(acc_mapping, k)


def pass_at_k_mapping(accmapping: Dict[str, Any], k) -> float:
    res = {}
    for key, value in accmapping.items():
        if len(value) % k != 0:
            return None  # Invalid pass at k
        res[key] = np.mean([max(value[i : (i + k)]) for i in range(0, len(value), k)])

    return np.mean(list(res.values()))
