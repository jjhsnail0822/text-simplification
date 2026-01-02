#!/bin/bash
#SBATCH -q cpu_qos
#SBATCH -p dell_cpu

set -euo pipefail
mkdir -p logs results/evaluation

# Define output directory
OUT_DIR="results/evaluation"

# echo "Evaluating Original Text Coherence..."
# python src/experiments/evaluate_coherence_api.py \
#     --mode original \
#     --output_file $OUT_DIR/eval_original_coherence.json

ZERO_SHOT_FILE="results/llm_test/Qwen3-4B-Instruct-2507.json"
if [ -f "$ZERO_SHOT_FILE" ]; then
    echo "Evaluating Zero-Shot Coherence..."
    python src/experiments/evaluate_coherence_api.py \
        --mode zero_shot \
        --input_file $ZERO_SHOT_FILE \
        --output_file $OUT_DIR/eval_zeroshot_coherence.json
else
    echo "Skipping Zero-Shot (file not found)"
fi

GPT_FILE="results/llm_test/gpt-5.2.json"
if [ -f "$GPT_FILE" ]; then
    echo "Evaluating GPT-5.2 Coherence..."
    python src/experiments/evaluate_coherence_api.py \
        --mode zero_shot \
        --input_file $GPT_FILE \
        --output_file $OUT_DIR/eval_gpt52_coherence.json
else
    echo "Skipping GPT-5.2 (file not found)"
fi

# GEMINI_FILE="results/llm_test/gemini-2.5-flash.json"
# if [ -f "$GEMINI_FILE" ]; then
#     echo "Evaluating GEMINI 2.5 Coherence..."
#     python src/experiments/evaluate_coherence_api.py \
#         --mode zero_shot \
#         --input_file $GEMINI_FILE \
#         --output_file $OUT_DIR/eval_gemini25_coherence.json
# else
#     echo "Skipping GEMINI 2.5 (file not found)"
# fi

# CLAUDE_FILE="results/llm_test/claude-sonnet-4-5-20250929.json"
# if [ -f "$CLAUDE_FILE" ]; then
#     echo "Evaluating Claude 4.5 Coherence..."
#     python src/experiments/evaluate_coherence_api.py \
#         --mode zero_shot \
#         --input_file $CLAUDE_FILE \
#         --output_file $OUT_DIR/eval_claude45_coherence.json
# else
#     echo "Skipping Claude 4.5 (file not found)"
# fi

if [ -f "data/fudge_results.json" ]; then
    echo "Evaluating FUDGE Coherence..."
    python src/experiments/evaluate_coherence_api.py \
        --mode fudge \
        --input_file data/fudge_results.json \
        --output_file $OUT_DIR/eval_fudge_coherence.json
else
    echo "Skipping FUDGE (file not found)"
fi

TRAINED_FILE="results/trained_llm_test/checkpoint-2596-merged.json"
if [ -f "$TRAINED_FILE" ]; then
    echo "Evaluating Trained Model Coherence..."
    python src/experiments/evaluate_coherence_api.py \
        --mode trained \
        --input_file $TRAINED_FILE \
        --output_file $OUT_DIR/eval_trained_coherence.json
else
    echo "Skipping Trained Model (file not found)"
fi

echo "Coherence evaluation completed."