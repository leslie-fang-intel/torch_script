# trt llm version: 2d2b8bae32b1d65f44873d762eb44fbc38e1336d
"""DeepSeek-V3-Lite ONLINE EPLB test script.

Online EPLB collects expert routing statistics at runtime via the EPLB
pipeline (start_wait_gpu_stage / update_statistic / route / set_cpu_stage),
asks the CPU workerThread to recompute placement periodically, and migrates
expert weights between GPU slots via cudaMemcpy.

Differences vs offline (test_dsv3lite_offline_eplb.py):
  - layer_updates_per_iter > 0   (offline: 0)
  - initial_global_assignments OMITTED  (runtime computes/overrides it)
  - statistics collected by runtime EPLB pipeline (no offline analysis needed)
  - CPU workerThread + computeAndUpdateThread run in the background
  - GDRCopy is REQUIRED (for host-accessible device memory)

Tunable via environment variables:
  NUM_SLOTS=80                  (= 72 experts + 8 redundancy; must be % ep_size == 0)
  LAYER_UPDATES_PER_ITER=2      (how many MoE layers updated per iter)
  MAX_TOKENS=128
  PROMPTS_FILE=./prompts.txt

Usage:
  python test_dsv3lite_online_eplb.py

To verify online EPLB is actually balancing load, also collect runtime stats:
  EXPERT_STATISTIC_PATH=./expert_statistic_online \
  EXPERT_STATISTIC_ITER_RANGE=10-100 \
  PROMPTS_FILE=./prompts.txt MAX_TOKENS=128 \
  python test_dsv3lite_online_eplb.py
Then compare against the no-EPLB baseline with:
  python ./examples/wide_ep/ep_load_balancer/report_load_statistics.py \
         --expert_statistic_path ./expert_statistic_online
"""
import os

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import MoeConfig


def load_prompts():
    prompts_file = os.environ.get("PROMPTS_FILE")
    if prompts_file and os.path.isfile(prompts_file):
        with open(prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"[INFO] loaded {len(prompts)} prompts from {prompts_file}",
              flush=True)
        return prompts
    return [
        "Describe the moon in one sentence.",
        "Describe Python in one sentence.",
        "Describe TRT-LLM in one sentence.",
        "Describe MoE models in one sentence.",
        "Describe CUDA graphs in one sentence.",
        "Describe FP8 in one sentence.",
        "Describe tensor parallelism in one sentence.",
        "Describe attention DP in one sentence.",
    ]


def build_online_moe_config():
    """Construct online EPLB config inline (no yaml needed).

    Online EPLB does not need ``initial_global_assignments`` — the runtime
    overrides placement on every iter end based on accumulated statistics.
    """
    num_slots = int(os.environ.get("NUM_SLOTS", "80"))
    layer_updates_per_iter = int(
        os.environ.get("LAYER_UPDATES_PER_ITER", "2"))
    assert layer_updates_per_iter > 0, (
        "Online EPLB requires layer_updates_per_iter > 0. "
        "Use test_dsv3lite_offline_eplb.py for static (offline) placement.")
    print(
        f"[INFO] online EPLB: num_slots={num_slots}, "
        f"layer_updates_per_iter={layer_updates_per_iter}",
        flush=True)
    # Pass as dict — MoeConfig.load_balancer accepts Union[MoeLoadBalancerConfig, dict, str]
    # and pydantic will validate/construct MoeLoadBalancerConfig internally.
    return MoeConfig(load_balancer={
        "num_slots": num_slots,
        "layer_updates_per_iter": layer_updates_per_iter,
    })


def main():
    MODEL_PATH = "/llm-models/DeepSeek-V3-Lite/fp8"
    max_tokens = int(os.environ.get("MAX_TOKENS", "128"))

    prompts = load_prompts()
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=max_tokens)

    moe_config = build_online_moe_config()

    print("---- start to create llm ----", flush=True)
    with LLM(
            model=MODEL_PATH,
            tensor_parallel_size=4,
            moe_expert_parallel_size=4,
            enable_attention_dp=True,
            moe_config=moe_config,
    ) as llm:
        print("\n ---- start llm.generate ---- \n", flush=True)
        outputs = llm.generate(prompts, sampling_params)
        print(f"---- finish generate ({len(outputs)} outputs) ----",
              flush=True)

        for output in outputs[:8]:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(
                f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# The entry point of the program need to be protected for spawning processes.
if __name__ == '__main__':
    main()
