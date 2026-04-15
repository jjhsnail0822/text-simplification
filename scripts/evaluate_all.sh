set -euo pipefail
mkdir -p logs results/evaluation

export MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
# Evaluator Models
export EVALUATOR_MODEL_1="google/gemma-3-27b-it"
export EVALUATOR_MODEL_2="Qwen/Qwen3-32B"

export USE_EVAL_VLLM=1
export EVAL_VLLM_ENDPOINT=http://localhost:8008/v1

export WEIGHT_VOCAB=2.0
export WEIGHT_ENTAILMENT=1.0
export WEIGHT_COHERENCE=1.0

# Function to manage vLLM and run evaluation in two phases
run_evaluation() {
    MODE=$1
    INPUT_FILE=$2
    OUTPUT_FILE=$3
    TEMP_FILE="${OUTPUT_FILE}.temp"

    echo "========================================================"
    echo "Starting Evaluation for Mode: $MODE"
    echo "Output: $OUTPUT_FILE"
    echo "========================================================"

    if [ -f "$TEMP_FILE" ]; then
        echo "Temp file $TEMP_FILE found. Skipping Phase 1."
    else
        
        # ---------------------------------------------------------
        # Phase 1: Compute Vocab, Entailment, Coherence (Model 1)
        # ---------------------------------------------------------
        echo "[Phase 1] Starting vLLM with $EVALUATOR_MODEL_1 on GPUs 0,1..."
        CUDA_VISIBLE_DEVICES=0,1 vllm serve $EVALUATOR_MODEL_1 \
        --port 8008 --data-parallel-size 1 --tensor-parallel-size 2 --gpu-memory-utilization 0.95 \
        --served-model-name evaluator \
        --max-model-len 4096 \
        --uvicorn-log-level warning \
        --disable-uvicorn-access-log \
        --disable-log-requests \
        > logs/vllm_eval_p1-$SLURM_JOB_ID.log 2>&1 &
        VLLM_PID=$!

        # Wait for vLLM
        count=0
        until curl -sSf http://localhost:8008/health >/dev/null; do
        echo "Waiting for vLLM (Model 1) to be ready... ($count s)"
        sleep 30
        count=$((count+30))
        if [ $count -ge 1200 ]; then echo "vLLM timeout"; kill $VLLM_PID; exit 1; fi
        done

        echo "[Phase 1] Running evaluation script..."
        # Export the correct model ID for Phase 1 so python script knows what to use
        export EVALUATOR_MODEL_ID=$EVALUATOR_MODEL_1
        
        # Use GPU 2 for the python script
        CUDA_VISIBLE_DEVICES=2 python src/experiments/evaluate_all.py \
            --mode "$MODE" \
            --input_file "$INPUT_FILE" \
            --output_file "$OUTPUT_FILE" \
            --temp_file "$TEMP_FILE" \
            --phase 1 \
            --model_id "$MODEL_ID" \

        echo "[Phase 1] Stopping vLLM..."
        kill $VLLM_PID
        wait $VLLM_PID
        sleep 10
    fi

    # ---------------------------------------------------------
    # Phase 2: Compute Coherence (Model 2) and Average
    # ---------------------------------------------------------
    echo "[Phase 2] Starting vLLM with $EVALUATOR_MODEL_2 on GPUs 0,1..."
    CUDA_VISIBLE_DEVICES=0,1 vllm serve $EVALUATOR_MODEL_2 \
      --port 8008 --data-parallel-size 1 --tensor-parallel-size 2 --gpu-memory-utilization 0.95 \
      --served-model-name evaluator \
      --max-model-len 4096 \
      --uvicorn-log-level warning \
      --disable-uvicorn-access-log \
      --disable-log-requests \
      > logs/vllm_eval_p2-$SLURM_JOB_ID.log 2>&1 &
    VLLM_PID=$!

    # Wait for vLLM
    count=0
    until curl -sSf http://localhost:8008/health >/dev/null; do
      echo "Waiting for vLLM (Model 2) to be ready... ($count s)"
      sleep 30
      count=$((count+30))
      if [ $count -ge 1200 ]; then echo "vLLM timeout"; kill $VLLM_PID; exit 1; fi
    done

    echo "[Phase 2] Running evaluation script..."
    # Export the correct model ID for Phase 2
    export EVALUATOR_MODEL_ID=$EVALUATOR_MODEL_2

    # Use GPU 2 for the python script
    CUDA_VISIBLE_DEVICES=2 python src/experiments/evaluate_all.py \
        --mode "$MODE" \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE" \
        --temp_file "$TEMP_FILE" \
        --phase 2 \
        --model_id "$MODEL_ID" \

    echo "[Phase 2] Stopping vLLM..."
    kill $VLLM_PID
    wait $VLLM_PID
    sleep 10
    
    # Cleanup temp file
    if [ -f "$TEMP_FILE" ]; then
        rm "$TEMP_FILE"
    fi
    
    echo "Evaluation for $MODE completed."
}

# ---------------------------------------------------------
# Run Evaluations
# ---------------------------------------------------------

# Zero-Shot
ZERO_SHOT_FILE="results/llm_test/Qwen3-4B-Instruct-2507.json"
if [ -f "$ZERO_SHOT_FILE" ]; then
    run_evaluation "zero_shot" "$ZERO_SHOT_FILE" "results/evaluation/eval_zeroshot_Qwen3-4B-Instruct-2507.json"
else
    echo "Skipping Zero-Shot (file not found)"
fi

ZERO_SHOT_FILE="results/llm_test/gemma-3-4b-it.json"
if [ -f "$ZERO_SHOT_FILE" ]; then
    run_evaluation "zero_shot" "$ZERO_SHOT_FILE" "results/evaluation/eval_zeroshot_gemma-3-4b-it.json"
else
    echo "Skipping Zero-Shot (file not found)"
fi

ZERO_SHOT_FILE="results/llm_test_pgv/Qwen3-4B-Instruct-2507.json"
if [ -f "$ZERO_SHOT_FILE" ]; then
    run_evaluation "zero_shot" "$ZERO_SHOT_FILE" "results/evaluation/eval_pgv_zeroshot_Qwen3-4B-Instruct-2507.json"
else
    echo "Skipping Zero-Shot (file not found)"
fi

ZERO_SHOT_FILE="results/llm_test_pgv/gemma-3-4b-it.json"
if [ -f "$ZERO_SHOT_FILE" ]; then
    run_evaluation "zero_shot" "$ZERO_SHOT_FILE" "results/evaluation/eval_pgv_zeroshot_gemma-3-4b-it.json"
else
    echo "Skipping Zero-Shot (file not found)"
fi

TRAINED_FILE="results/llm_test/Qwen3-4B-Instruct-2507-trained.json"
if [ -f "$TRAINED_FILE" ]; then
    run_evaluation "trained" "$TRAINED_FILE" "results/evaluation/eval_Qwen3-4B-Instruct-2507-trained.json"
else
    echo "Skipping Trained Model (file not found)"
fi

TRAINED_FILE="results/llm_test_pgv/Qwen3-4B-Instruct-2507-trained.json"
if [ -f "$TRAINED_FILE" ]; then
    run_evaluation "trained" "$TRAINED_FILE" "results/evaluation/eval_pgv_Qwen3-4B-Instruct-2507-trained.json"
else
    echo "Skipping Trained Model (file not found)"
fi

TRAINED_FILE="results/llm_test/gemma-3-4b-it-trained.json"
if [ -f "$TRAINED_FILE" ]; then
    run_evaluation "trained" "$TRAINED_FILE" "results/evaluation/eval_gemma-3-4b-it-trained.json"
else
    echo "Skipping Trained Model (file not found)"
fi

TRAINED_FILE="results/llm_test_pgv/gemma-3-4b-it-trained.json"
if [ -f "$TRAINED_FILE" ]; then
    run_evaluation "trained" "$TRAINED_FILE" "results/evaluation/eval_pgv_gemma-3-4b-it-trained.json"
else
    echo "Skipping Trained Model (file not found)"
fi

# GPT-5.2
GPT_FILE="results/llm_test/gpt-5.2.json"
if [ -f "$GPT_FILE" ]; then
    run_evaluation "zero_shot" "$GPT_FILE" "results/evaluation/eval_gpt52.json"
else
    echo "Skipping GPT-5.2 (file not found)"
fi

# Gemini 2.5
GEMINI_FILE="results/llm_test/gemini-2.5-flash.json"
if [ -f "$GEMINI_FILE" ]; then
    run_evaluation "zero_shot" "$GEMINI_FILE" "results/evaluation/eval_gemini25.json"
else
    echo "Skipping GEMINI (file not found)"
fi

# FUDGE
if [ -f "data/fudge_results.json" ]; then
    run_evaluation "fudge" "data/fudge_results.json" "results/evaluation/eval_fudge.json"
else
    echo "Skipping FUDGE (file not found)"
fi

echo "All evaluations completed."
