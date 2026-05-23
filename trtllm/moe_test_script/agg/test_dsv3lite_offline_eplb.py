# trt llm version: 2d2b8bae32b1d65f44873d762eb44fbc38e1336d
"""DeepSeek-V3-Lite offline EPLB test script.

Three modes controlled by environment variables:

  1. Plain inference (no env vars):
     python test_dsv3lite_offline_eplb.py

  2. Statistic collection (Stage 1 of offline EPLB):
     EXPERT_STATISTIC_PATH=./expert_statistic \
     EXPERT_STATISTIC_ITER_RANGE=10-200 \
     PROMPTS_FILE=./prompts.txt MAX_TOKENS=128 \
     python test_dsv3lite_offline_eplb.py

  3. Offline EPLB deployment (Stage 3 of offline EPLB):
     MOE_LB_CONFIG=$(pwd)/moe_load_balancer.yaml \
     PROMPTS_FILE=./prompts.txt MAX_TOKENS=128 \
     python test_dsv3lite_offline_eplb.py
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


def maybe_build_moe_config():
    yaml_path = os.environ.get("MOE_LB_CONFIG")
    if yaml_path:
        assert os.path.isfile(
            yaml_path), f"MOE_LB_CONFIG not found: {yaml_path}"
        print(f"[INFO] using offline EPLB config: {yaml_path}", flush=True)
        return MoeConfig(load_balancer=yaml_path)
    return None


def main():
    MODEL_PATH = "/llm-models/DeepSeek-V3-Lite/fp8"
    max_tokens = int(os.environ.get("MAX_TOKENS", "128"))

    prompts = load_prompts()
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=max_tokens)

    moe_config = maybe_build_moe_config()
    llm_kwargs = dict(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        moe_expert_parallel_size=4,
        enable_attention_dp=True,
    )
    if moe_config is not None:
        llm_kwargs["moe_config"] = moe_config

    print("---- start to create llm ----", flush=True)
    with LLM(**llm_kwargs) as llm:
        print("\n ---- start llm.generate ---- \n", flush=True)
        outputs = llm.generate(prompts, sampling_params)
        print(f"---- finish generate ({len(outputs)} outputs) ----",
              flush=True)

        # Only print first few to avoid flooding the log
        for output in outputs[:8]:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(
                f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# The entry point of the program need to be protected for spawning processes.
if __name__ == '__main__':
    main()
