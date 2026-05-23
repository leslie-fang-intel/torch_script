# Disaggregated MoE Test Scripts

DeepSeek-V3-Lite tests using **disaggregated serving** on 1 node × 4 GPU
(no SLURM required). Prefill (context) and decode (generation) run on
separate worker pools; KV cache is exchanged via UCX.

## Layout (default)

```
GPU 0,1  → ctx server (TP=2 EP=2 attention_dp)   :8001
GPU 2,3  → gen server (TP=2 EP=2 attention_dp)   :8003
CPU      → disagg router                          :8000   ← clients connect here
```

## Files

| File | Role |
|---|---|
| `ctx_config.yaml` | Extra LLM API options for context (prefill) workers |
| `gen_config.yaml` | Extra LLM API options for generation (decode) workers |
| `disagg_server.yaml` | Router config listing ctx/gen worker URLs |
| `launch.sh` | Start all 3 servers in background, wait for ready |
| `shutdown.sh` | Stop all 3 servers cleanly |
| `send_request.sh` | Send a test prompt via curl |

## Quick start (baseline, no EPLB)

Run from TRT-LLM root:

```bash
cd /workspace/TensorRT-LLM

# 1. Launch all 3 servers (logs in ./disagg_logs/)
bash moe_test_script/disagg/launch.sh

# 2. Send a test request once all servers report ready
bash moe_test_script/disagg/send_request.sh

# 3. When done, tear everything down
bash moe_test_script/disagg/shutdown.sh
```

`launch.sh` blocks until all 3 servers respond on `/health`. First boot
takes ~1-2 minutes (loading FP8 weights twice — once per pool).

## Offline EPLB on gen pool only

Gen is where EPLB pays off most: decode iters dominate, routing pattern is
stable, expert imbalance directly hurts ITL. Ctx is compute-bound on
attention (long sequences); MoE imbalance is less impactful on prefill.

Three-stage workflow:

### Stage 1: collect routing stats from a baseline gen run

```bash
cd /workspace/TensorRT-LLM
rm -rf ./expert_statistic_gen        # clean residue
rm -f ./moe_load_balancer_gen.yaml

# Generate fake prompts (if not done already)
python moe_test_script/gen_fake_prompts.py     # writes ./prompts.txt

# Launch disagg with stats collection enabled ONLY on gen worker
GEN_EXPERT_STATISTIC_PATH=./expert_statistic_gen \
GEN_EXPERT_STATISTIC_ITER_RANGE=10-100 \
bash moe_test_script/disagg/launch.sh

# Drive workload through the disagg router so gen accumulates iters
python moe_test_script/disagg/send_prompts.py

# When done, shut down — gen worker dumps stats to expert_statistic_gen/
# at iter 100 (need to make sure traffic has actually pushed gen past iter 100)
bash moe_test_script/disagg/shutdown.sh

# Verify
ls -la ./expert_statistic_gen/
cat ./expert_statistic_gen/meta_info.json
```

You should see `meta_info.json` + `rank{0,1}.safetensors` (only 2 ranks
because gen has EP=2). If the files are missing, gen didn't reach iter
100 — increase `MAX_TOKENS` when sending prompts (env var of `send_prompts.py`)
or lower the stop value (e.g. `GEN_EXPERT_STATISTIC_ITER_RANGE=10-50`).

### Stage 2: generate placement yaml for gen

```bash
python ./examples/wide_ep/ep_load_balancer/generate_eplb_config.py \
    --ep_size 2 \
    --num_slots 80 \
    --expert_statistic_path ./expert_statistic_gen \
    --output_path ./moe_load_balancer_gen.yaml \
    --layer_updates_per_iter 0

head -20 ./moe_load_balancer_gen.yaml
```

Note `--ep_size 2` (not 4) — the disagg gen pool has EP=2. The yaml has
80 entries per layer (`num_slots`), and `slot_id / (80/2=40) = ep_rank`.

### Stage 3: redeploy with gen using the placement yaml

```bash
GEN_CONFIG=$(pwd)/moe_test_script/disagg/gen_config_with_eplb.yaml \
bash moe_test_script/disagg/launch.sh

# Send a test request — gen now uses the offline placement, ctx unchanged
bash moe_test_script/disagg/send_request.sh

# Shut down when done
bash moe_test_script/disagg/shutdown.sh
```

Verify in `disagg_logs/gen.log` you should see (only for gen, not ctx):
- `[INFO] using offline EPLB config: .../moe_load_balancer_gen.yaml`
- `Created MoE LoadBalancer, layer_updates_per_iter=0...`
- `MoE load balancer enabled. num_experts = 72, num_slots = 80, ep_size = 2`

### Stage 3+ (optional): verify balance improvement

Re-collect stats while gen runs with the new placement, then compare:

```bash
rm -rf ./expert_statistic_gen_after

GEN_CONFIG=$(pwd)/moe_test_script/disagg/gen_config_with_eplb.yaml \
GEN_EXPERT_STATISTIC_PATH=./expert_statistic_gen_after \
GEN_EXPERT_STATISTIC_ITER_RANGE=10-100 \
bash moe_test_script/disagg/launch.sh
python moe_test_script/disagg/send_prompts.py
bash moe_test_script/disagg/shutdown.sh

# Compare imbalance: before vs after
python ./examples/wide_ep/ep_load_balancer/report_load_statistics.py \
    --expert_statistic_path ./expert_statistic_gen
echo "---"
python ./examples/wide_ep/ep_load_balancer/report_load_statistics.py \
    --expert_statistic_path ./expert_statistic_gen_after
```

Expect lower `imbalance-ratio` in the "after" report.

## Enabling EPLB on ctx (uncommon, only if profiling shows ctx MoE imbalance hurts TTFT)

Edit `ctx_config.yaml`, uncomment the `moe_config` block, supply your own
ctx-specific placement yaml. Same `num_slots % ep_size == 0` rule applies.

## Troubleshooting

- **`trtllm-serve` not found**: install/activate the trtllm wheel
  (`pip install -e .` from repo root if running from source).
- **`address already in use`**: a previous launch left workers running.
  Run `shutdown.sh`, then `pkill -f trtllm-serve` if necessary.
- **`UCX_*` errors**: try setting `UCX_TLS=^cuda_ipc,tcp` or switch to
  `cache_transceiver_config.backend: DEFAULT` (NIXL fallback).
- **Stuck at `waiting for ctx :8001 ...`**: check `disagg_logs/ctx.log`
  for the actual worker error (model loading, OOM, etc).
- **Router 5xx**: check `disagg_logs/router.log` and worker logs;
  KV cache transfer failure usually shows up here.
