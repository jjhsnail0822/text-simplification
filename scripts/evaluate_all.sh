#!/bin/bash
#SBATCH -q base_qos
#SBATCH -p gigabyte_A6000
#SBATCH --gres=gpu:2

set -euo pipefail
mkdir -p logs results/evaluation

export MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
export EVALUATOR_MODEL_ID="Qwen/Qwen3-14B"
export USE_EVAL_VLLM=1
export EVAL_VLLM_ENDPOINT=http://localhost:8008/v1

export WEIGHT_VOCAB=2.0
export WEIGHT_ENTAILMENT=1.0
export WEIGHT_COHERENCE=2.0

echo "Starting vLLM server for evaluator..."
CUDA_VISIBLE_DEVICES=0 vllm serve $EVALUATOR_MODEL_ID \
  --port 8008 --data-parallel-size 1 --gpu-memory-utilization 0.95 \
  --served-model-name evaluator \
  --max-model-len 4096 \
  --uvicorn-log-level warning \
  --disable-uvicorn-access-log \
  --disable-log-requests \
  > logs/vllm_eval-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!

until curl -sSf http://localhost:8008/health >/dev/null; do
  echo "Waiting for vLLM to be ready..."
  sleep 10
done

export CUDA_VISIBLE_DEVICES=1

# echo "Evaluating Original Text..."
# python src/experiments/evaluate_all.py \
#     --mode original \
#     --output_file results/evaluation/eval_original.json \
#     --model_id $MODEL_ID

# ZERO_SHOT_FILE="results/llm_test/Qwen3-4B-Instruct-2507.json"
# if [ -f "$ZERO_SHOT_FILE" ]; then
#     echo "Evaluating Zero-Shot..."
#     python src/experiments/evaluate_all.py \
#         --mode zero_shot \
#         --input_file $ZERO_SHOT_FILE \
#         --output_file results/evaluation/eval_zeroshot.json \
#         --model_id $MODEL_ID
# else
#     echo "Skipping Zero-Shot (file not found at $ZERO_SHOT_FILE)"
# fi

# if [ -f "data/fudge_results.json" ]; then
#     echo "Evaluating FUDGE..."
#     python src/experiments/evaluate_all.py \
#         --mode fudge \
#         --input_file data/fudge_results.json \
#         --output_file results/evaluation/eval_fudge.json \
#         --model_id $MODEL_ID
# else
#     echo "Skipping FUDGE (file not found)"
# fi

TRAINED_FILE="results/trained_llm_test/checkpoint-1600-merged.json"
if [ -f "$TRAINED_FILE" ]; then
    echo "Evaluating Trained Model..."
    python src/experiments/evaluate_all.py \
        --mode trained \
        --input_file $TRAINED_FILE \
        --output_file results/evaluation/eval_trained.json \
        --model_id $MODEL_ID
else
    echo "Skipping Trained Model (file not found at $TRAINED_FILE)"
fi

echo "Stopping vLLM..."
kill $VLLM_PID
wait $VLLM_PID
echo "Evaluation completed."