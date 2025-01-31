#!/bin/bash
export NAME=volcano
export version=12.4
conda create -n $NAME python=3.10 -y
conda activate $NAME
conda install pytorch torchvision torchaudio pytorch-cuda=$version -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-${version}.0" cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX

pip install flash-attn --no-build-isolation
pip install vllm sympy regex latex2sympy2 word2nubmer
pip install -e .