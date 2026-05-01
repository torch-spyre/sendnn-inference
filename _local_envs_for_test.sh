#!/bin/bash

# Env vars to set for CPU-only testing

# Need to be set for tests to run
export MASTER_ADDR=localhost
export MASTER_PORT=12345

# Run on CPU
export SENDNN_INFERENCE_DYNAMO_BACKEND=eager

# Test related

# This makes debugging easy on local setup
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# We have to use `HF_HUB_OFFLINE=1` otherwise vllm might try to download a
# different version of the model using HF API which might not work locally
export HF_HUB_OFFLINE=1
