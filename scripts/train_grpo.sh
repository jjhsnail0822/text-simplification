#!/bin/bash
#SBATCH -q base_qos
#SBATCH -p gigabyte_A6000
#SBATCH --gres=gpu:4

set -euo pipefail
mkdir -p logs

export MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
export EVALUATOR_MODEL_ID="Qwen/Qwen3-14B"

export OUTPUT_DIR="results/grpo/Qwen3-4B-Instruct-2507-1.0-1.0-1.0-ablation"

export WEIGHT_VOCAB=1.0
export WEIGHT_ENTAILMENT=1.0
export WEIGHT_COHERENCE=1.0

export TRAINING_FLAG=1

export USE_EVAL_VLLM=1
export VLLM_BATCH_INVARIANT=1
export EVAL_VLLM_ENDPOINT=http://localhost:8008/v1

CUDA_VISIBLE_DEVICES=0 vllm serve $EVALUATOR_MODEL_ID \
  --port 8008 --data-parallel-size 1 --gpu-memory-utilization 0.95 \
  --served-model-name evaluator \
  --max-model-len 4096 \
  --uvicorn-log-level warning \
  --disable-uvicorn-access-log \
  --disable-log-requests \
  > logs/vllm-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!

until curl -sSf http://localhost:8008/health >/dev/null; do
  echo "Waiting for vLLM to be ready..."
  sleep 30
done

export CUDA_VISIBLE_DEVICES=1,2,3

echo "Starting training..."
accelerate launch --num_processes 3 src/experiments/train_grpo.py

kill $VLLM_PID
wait $VLLM_PID