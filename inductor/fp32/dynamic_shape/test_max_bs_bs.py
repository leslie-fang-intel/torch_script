import torch
import torch.nn as nn
import time

# CMD: TORCHINDUCTOR_FREEZING=1 TORCH_LOGS="+output_code" numactl -C 56-56 --membind=1 python test_kv_cache_code_gen.py 2>&1 | tee ut_test.log

# Structure refers to 
# https://github.com/pytorch/benchmark/blob/79c236aed69907988b941e730965e6bfc9fd8c21/torchbenchmark/models/llama/model.py#L124-L133

class M(nn.Module):
    def __init__(self, update_kv_cache=True):
        super(M, self).__init__()
        self.linear1 = nn.Linear(64, 64, bias=False)
        max_bs = 32
        self.cache_k = torch.zeros((max_bs, 384, 8, 64), device="cpu")
        self.update_kv_cache = update_kv_cache

    def forward(self, x, start_pos):
        bsz, seqlen, _, _ = x.shape
        xk = self.linear1(x)
        if self.update_kv_cache:
            with torch.no_grad():
                self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        scores = torch.matmul(xk.transpose(1, 2), keys.transpose(1, 2).transpose(2, 3))
        return scores

def test_kv_cache():
    kv_cache_module = M()
    # kv_cache_module = M(update_kv_cache=False)
    input = torch.randn(32, 32, 8, 64)

    with torch.no_grad():

        compiled_model = torch.compile(kv_cache_module)

        torch._dynamo.mark_dynamic(input, 0)

        compiled_model(input, 1)
        compiled_model(input, 1)

if __name__ == "__main__":
    test_kv_cache()
