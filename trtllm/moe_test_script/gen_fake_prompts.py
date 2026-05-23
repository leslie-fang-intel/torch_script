#!/usr/bin/env python
"""Generate a fake multi-topic prompts file for offline EPLB statistic collection.

Output: ./prompts.txt (one prompt per line).
"""
import random


def main():
    random.seed(42)
    topics = [
        "moon", "Python", "TensorRT-LLM", "MoE models", "CUDA graphs",
        "FP8 quantization", "tensor parallelism", "attention DP",
        "expert parallelism", "KV cache", "FlashAttention",
        "speculative decoding", "MLA", "deepseek-v3", "rotary embeddings",
        "AlltoAll communication", "load balancer", "prefill phase",
        "decode phase", "transformer architecture", "RoPE", "RMSNorm",
        "SwiGLU activation", "softmax routing", "topK routing",
        "pipeline parallelism", "NCCL collectives", "NVLink",
        "ProgrammaticDependentLaunch", "GDRCopy", "shared memory",
        "global memory", "atomic operations", "warp shuffle", "tensor cores",
        "MMA instruction", "GEMM kernel", "FP16 vs BF16",
        "INT4 weight only", "AWQ quantization", "ReduceScatter",
    ]
    templates = [
        "Describe {x} in one sentence.",
        "What is {x}?",
        "Explain {x} briefly.",
        "Compare {x} with alternatives.",
        "List the pros and cons of {x}.",
        "How does {x} work internally?",
        "Why is {x} important for inference?",
        "Walk me through {x}.",
        "Summarize {x} for a beginner.",
        "Give an example of {x}.",
    ]

    prompts = []
    for _ in range(3):
        for t in topics:
            prompts.append(random.choice(templates).format(x=t))
    random.shuffle(prompts)

    out_path = "./prompts.txt"
    with open(out_path, "w") as f:
        for p in prompts:
            f.write(p + "\n")
    print(f"Wrote {len(prompts)} prompts to {out_path}")


if __name__ == "__main__":
    main()
