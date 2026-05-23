#!/usr/bin/env bash
# Clean up the 3 servers started by launch.sh.
#
# Run from TensorRT-LLM root:
#   bash moe_test_script/disagg/shutdown.sh

LOG_DIR="${LOG_DIR:-./disagg_logs}"

kill_pid() {
    local name=$1
    local pidfile="$LOG_DIR/${name}.pid"
    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[INFO] killing $name (PID=$pid) and children"
            # kill the process group so MPI workers and ZMQ subprocs also go
            pkill -P "$pid" 2>/dev/null
            kill "$pid" 2>/dev/null
            sleep 1
            kill -9 "$pid" 2>/dev/null
        else
            echo "[INFO] $name (PID=$pid) not running"
        fi
        rm -f "$pidfile"
    else
        echo "[INFO] no $name.pid; skipping"
    fi
}

# Kill in reverse launch order: router first, then workers
kill_pid router
kill_pid gen
kill_pid ctx

# Belt-and-suspenders: stragglers
pkill -f "trtllm-serve disaggregated" 2>/dev/null
pkill -f "trtllm-serve.*--port 8001" 2>/dev/null
pkill -f "trtllm-serve.*--port 8003" 2>/dev/null

echo "[INFO] shutdown complete"
