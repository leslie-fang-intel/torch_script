#!/usr/bin/env bash
# Launch DSv3-Lite disaggregated serving on 1 node × 4 GPU.
#
# Layout:
#   GPU 0,1  → ctx server (TP=2 EP=2 ADP), port 8001
#   GPU 2,3  → gen server (TP=2 EP=2 ADP), port 8003
#   CPU      → disagg router/proxy,         port 8000
#
# Run from TensorRT-LLM root:
#   bash moe_test_script/disagg/launch.sh
#
# Optional env vars (for offline EPLB workflow on gen only):
#   GEN_CONFIG=/abs/path/to/gen_cfg.yaml          override default gen config
#   GEN_EXPERT_STATISTIC_PATH=./expert_statistic_gen     enable stat collection on gen worker
#   GEN_EXPERT_STATISTIC_ITER_RANGE=10-100
#
# Logs go to ./disagg_logs/{ctx,gen,router}.log
# PIDs go to ./disagg_logs/*.pid for clean shutdown.

set -uo pipefail

MODEL_PATH="${MODEL_PATH:-/llm-models/DeepSeek-V3-Lite/fp8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-./disagg_logs}"
mkdir -p "$LOG_DIR"

CTX_CONFIG="${CTX_CONFIG:-$SCRIPT_DIR/ctx_config.yaml}"
GEN_CONFIG="${GEN_CONFIG:-$SCRIPT_DIR/gen_config.yaml}"
ROUTER_CONFIG="${ROUTER_CONFIG:-$SCRIPT_DIR/disagg_server.yaml}"

# Sanity check
for f in "$CTX_CONFIG" "$GEN_CONFIG" "$ROUTER_CONFIG"; do
    [[ -f "$f" ]] || { echo "[ERROR] missing config: $f"; exit 1; }
done
echo "[INFO] ctx config:    $CTX_CONFIG"
echo "[INFO] gen config:    $GEN_CONFIG"
echo "[INFO] router config: $ROUTER_CONFIG"

# Optional: inject EPLB stats collection ONLY on gen worker.
# Set via env vars; we pass them through to trtllm-serve so they reach the
# gen worker subprocess but NOT the ctx worker.
GEN_STAT_ENVS=()
if [[ -n "${GEN_EXPERT_STATISTIC_PATH:-}" ]]; then
    GEN_STAT_ENVS+=("EXPERT_STATISTIC_PATH=$GEN_EXPERT_STATISTIC_PATH")
fi
if [[ -n "${GEN_EXPERT_STATISTIC_ITER_RANGE:-}" ]]; then
    GEN_STAT_ENVS+=("EXPERT_STATISTIC_ITER_RANGE=$GEN_EXPERT_STATISTIC_ITER_RANGE")
fi
if (( ${#GEN_STAT_ENVS[@]} > 0 )); then
    echo "[INFO] gen worker will collect EPLB stats: ${GEN_STAT_ENVS[*]}"
fi

echo "[INFO] launching ctx server on GPU 0,1 :8001"
CUDA_VISIBLE_DEVICES=0,1 trtllm-serve "$MODEL_PATH" \
    --host localhost --port 8001 \
    --backend pytorch \
    --tp_size 2 --ep_size 2 \
    --enable_attention_dp \
    --config "$CTX_CONFIG" \
    &> "$LOG_DIR/ctx.log" &
echo $! > "$LOG_DIR/ctx.pid"
echo "[INFO]   ctx PID=$(cat "$LOG_DIR/ctx.pid")"

echo "[INFO] launching gen server on GPU 2,3 :8003"
# Use `env KEY=VAL ... CMD` to inject env vars from a bash array:
# quoted array expansion ("${arr[@]}") yields command arguments, not
# inline env assignments, so we need the `env` wrapper.
# Plain `env` (no `-i`) preserves PATH etc.
env CUDA_VISIBLE_DEVICES=2,3 "${GEN_STAT_ENVS[@]}" trtllm-serve "$MODEL_PATH" \
    --host localhost --port 8003 \
    --backend pytorch \
    --tp_size 2 --ep_size 2 \
    --enable_attention_dp \
    --config "$GEN_CONFIG" \
    &> "$LOG_DIR/gen.log" &
echo $! > "$LOG_DIR/gen.pid"
echo "[INFO]   gen PID=$(cat "$LOG_DIR/gen.pid")"

# Wait until both worker servers expose /health
wait_for_url() {
    local url=$1
    local timeout=${2:-600}
    local elapsed=0
    while (( elapsed < timeout )); do
        if curl -sf "$url" -o /dev/null 2>&1; then
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo "[INFO]   ($elapsed s) waiting for $url ..."
    done
    echo "[ERROR] timed out waiting for $url"
    return 1
}

echo "[INFO] waiting for ctx :8001 and gen :8003 ..."
wait_for_url "http://localhost:8001/health" 600 || { echo "[ERROR] ctx not ready"; exit 1; }
wait_for_url "http://localhost:8003/health" 600 || { echo "[ERROR] gen not ready"; exit 1; }

echo "[INFO] launching disagg router :8000"
trtllm-serve disaggregated -c "$ROUTER_CONFIG" \
    &> "$LOG_DIR/router.log" &
echo $! > "$LOG_DIR/router.pid"
echo "[INFO]   router PID=$(cat "$LOG_DIR/router.pid")"

wait_for_url "http://localhost:8000/health" 120 || { echo "[ERROR] router not ready"; exit 1; }

echo
echo "[INFO] disagg serving READY"
echo "[INFO]   Public endpoint:    http://localhost:8000"
echo "[INFO]   Logs:               $LOG_DIR/{ctx,gen,router}.log"
echo "[INFO]   Send a request:     bash $SCRIPT_DIR/send_request.sh"
echo "[INFO]   Replay prompts:     python $SCRIPT_DIR/send_prompts.py"
echo "[INFO]   Stop everything:    bash $SCRIPT_DIR/shutdown.sh"
