# Aggregated MoE Test Scripts

DeepSeek-V3-Lite tests using the LLM API in **aggregated** serving mode
(single Python process spawns TP/EP workers, no disaggregated prefill/decode
separation). All scripts use `MODEL_PATH=/llm-models/DeepSeek-V3-Lite/fp8`,
`tp=ep=4`, `enable_attention_dp=True`.

## Scripts

| Script | Purpose |
|---|---|
| `test_dsv3lite.py` | Plain baseline run (no EPLB, default parallelism config) |
| `test_dsv3lite_dtp.py` | DTP mode: attention DP + MoE TP (`moe_tp_size=4, moe_ep_size=1`) |
| `test_dsv3lite_tep.py` | TEP mode: attention TP + MoE EP variant |
| `test_dsv3lite_offline_eplb.py` | Offline EPLB: load placement from yaml (3-stage workflow) |
| `test_dsv3lite_online_eplb.py` | Online EPLB: dynamic placement updates at runtime (requires GDRCopy) |

## Running

**Always run from the TRT-LLM root directory.** The scripts use relative
paths (`./prompts.txt`, `./moe_load_balancer.yaml`, `./expert_statistic/`)
that are interpreted against the current working directory.

```bash
cd /workspace/TensorRT-LLM   # or wherever TRT-LLM is checked out
python moe_test_script/agg/test_dsv3lite_online_eplb.py
```

## EPLB workflow

For `test_dsv3lite_offline_eplb.py`, see the three-stage workflow in the
script's docstring. Prereqs:

1. Generate `prompts.txt`:
   ```bash
   python moe_test_script/gen_fake_prompts.py
   ```
2. Collect statistics (Stage 1):
   ```bash
   EXPERT_STATISTIC_PATH=./expert_statistic \
   EXPERT_STATISTIC_ITER_RANGE=10-100 \
   PROMPTS_FILE=./prompts.txt MAX_TOKENS=128 \
   python moe_test_script/agg/test_dsv3lite_offline_eplb.py
   ```
3. Generate placement yaml (Stage 2):
   ```bash
   python examples/wide_ep/ep_load_balancer/generate_eplb_config.py \
       --ep_size 4 --num_slots 80 \
       --expert_statistic_path ./expert_statistic \
       --output_path ./moe_load_balancer.yaml \
       --layer_updates_per_iter 0
   ```
4. Deploy with placement (Stage 3):
   ```bash
   unset EXPERT_STATISTIC_PATH EXPERT_STATISTIC_ITER_RANGE
   MOE_LB_CONFIG=$(pwd)/moe_load_balancer.yaml \
   PROMPTS_FILE=./prompts.txt MAX_TOKENS=128 \
   python moe_test_script/agg/test_dsv3lite_offline_eplb.py
   ```

For `test_dsv3lite_online_eplb.py`, no offline stages needed — just run:
```bash
PROMPTS_FILE=./prompts.txt MAX_TOKENS=128 \
python moe_test_script/agg/test_dsv3lite_online_eplb.py
```
