# This file contains util functions for checkpointing and loading.
import os
import glob
import torch


def find_latest_checkpoint(ckpt_dir):
    checkpoints = glob.glob(os.path.join(ckpt_dir, "global_step_*"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("_")[-1]))
    return checkpoints[-1]
