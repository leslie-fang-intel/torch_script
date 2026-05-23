# trt llm version: 2d2b8bae32b1d65f44873d762eb44fbc38e1336d
from tensorrt_llm import LLM, SamplingParams


def main():
    MODEL_PATH = "/llm-models/DeepSeek-V3-Lite/fp8"

    # prompts = [
    #     "Niuniu is my son, he likes playing fd 2048, and he is now playing it on a 4x4 grid. He wants to know the maximum number of points he can get ",
    # ]

    prompts = [
        "Describe the moon in one sentence.",
        "Describe Python in one sentence.",
        "Describe TRT-LLM in one sentence.",
        "Describe MoE models in one sentence.",
        "Describe CUDA graphs in one sentence.",
        "Describe FP8 in one sentence.",
        "Describe tensor parallelism in one sentence.",
        "Describe attention DP in one sentence.",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    print("---- start to create llm ----", flush=True)
    # TEP: attention uses TP, MoE uses EP
    #   tp_size=4, moe_ep_size=4, moe_tp_size=1 (default), attention_dp=False
    with LLM(
            model=MODEL_PATH,
            tensor_parallel_size=4,
            moe_expert_parallel_size=1,
            enable_attention_dp=True,
    ) as llm:
        print("\n ---- start llm.generate ---- \n", flush=True)
        outputs = llm.generate(prompts, sampling_params)
        print("---- finish generate ----", flush=True)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# The entry point of the program need to be protected for spawning processes.
if __name__ == '__main__':
    main()
