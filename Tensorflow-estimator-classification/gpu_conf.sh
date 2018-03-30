#!/bin/bash

export CUDA_HOME=/usr/local/cuda-8.0/
export LD_LIBRARY_PATH=/users/yash.p/cuda_version/cuda/lib64
PATH=/users/yash.p/cuda_version/cuda/bin:${PATH}
export PATH
export CUDA_VISIBLE_DEVICES=3

