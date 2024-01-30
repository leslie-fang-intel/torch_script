# TORCHINDUCTOR_FREEZING=1  TORCH_LOGS="+output_code" numactl -C 56-111 -m 1 python test_softmax.py
# * Eager time: 8.335479021072388
# * Inductor Baseline time: 16.34844946861267

# Modfied the generated kernel implementation
# Refer to: https://github.com/pytorch/pytorch/blob/da0635d17c8fc777010fc3a2c5efedfade499432/aten/src/ATen/native/cpu/SoftMaxKernel.cpp#L147-L211

# * Inductor optimize 1 (naive loop fusion) time: 11.380375862121582
# * <TODO> Inductor optimize 2 (naive loop fusion + eliminate reduant store/load) time:
#   * Define a tmp buf to do accum; May not easy to implement in Inductor CPP backend
# * best_performance_generated_code.py time: 3.912254571914674

import torch
import time
import random
import numpy as np

local_seed= 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.attn_dropout = torch.nn.Dropout(0.1)

    def forward(self, input, causal_mask):
        query, key, value = input.split(768, dim=2)
        query = _split_heads(query, 12, 64)
        key = _split_heads(key, 12, 64)
        value = _split_heads(value, 12, 64)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        # attn_weights:
        # size(4, 12, 1024, 1024)
        # stride(12582912, 1048576, 1024, 1)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)   

        attn_output = torch.matmul(attn_weights, value)
        
        
        return attn_output, attn_weights

def _split_heads(tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    
if __name__ == "__main__":

    with torch.no_grad():
        m = M().eval()
        input = torch.randn(4, 1024, 2304).to(torch.bfloat16)
        causal_mask = (torch.randint(0, 2, (1, 1, 1024, 1024))).to(torch.bool)

        m(input, causal_mask)

        warmup_steps = 100
        steps = 1000

        # Refer path
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            ref_res = m(input, causal_mask)

            for _ in range(warmup_steps):
                m(input, causal_mask)

            ref_start = time.time()
            for _ in range(steps):
                m(input, causal_mask)
            ref_end = time.time()

        # # Jit path
        # with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        #     jit_m = torch.jit.trace(m, (input, causal_mask)).eval()
        #     jit_m = torch.jit.freeze(jit_m)

        # jit_res = jit_m(input, causal_mask)
        # for _ in range(warmup_steps):
        #     jit_m(input, causal_mask)
        # print(jit_m.graph_for(input, causal_mask), flush=True)
        # jit_start = time.time()
        # for _ in range(steps):
        #     jit_m(input, causal_mask)
        # jit_end = time.time()

        # Compiler Path
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            c_m = torch.compile(m)
            inductor_res = c_m(input, causal_mask)

            for _ in range(warmup_steps):
                c_m(input, causal_mask)

            inductor_start = time.time()
            for _ in range(steps):
                c_m(input, causal_mask)
            inductor_end = time.time()
        print("ref time is: {}".format(ref_end - ref_start), flush=True)
        # print("jit time is: {}".format(jit_end - jit_start), flush=True)
        print("inductor time is: {}".format(inductor_end - inductor_start), flush=True)
        print(torch.allclose(ref_res[0], inductor_res[0], atol=0.01, rtol=0.01), flush=True)
        print(torch.allclose(ref_res[1], inductor_res[1], atol=0.01, rtol=0.01), flush=True)



