#!/usr/bin/env bash
# Send a test request to the disagg router.
#
# Run from TensorRT-LLM root after launch.sh finishes:
#   bash moe_test_script/disagg/send_request.sh

MODEL_PATH="${MODEL_PATH:-/llm-models/DeepSeek-V3-Lite/fp8}"
URL="${URL:-http://localhost:8000/v1/completions}"

echo "[INFO] POST $URL"
curl -sf "$URL" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_PATH\",
        \"prompt\": \"Describe MoE expert parallelism in one sentence.\",
        \"max_tokens\": 64,
        \"temperature\": 0.0
    }" | python -m json.tool
