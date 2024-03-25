
import torch
import torch._dynamo as torchdynamo
from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping,
)
from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
)
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import torch.nn as nn

def test_sdpa_pattern():
    import copy
    from torch import _dynamo, _inductor
    from torch._inductor import config
    import logging
    import numpy as np
    import random

    local_seed = 2023
    torch.manual_seed(local_seed) # Set PyTorch seed
    np.random.seed(seed=local_seed) # Set Numpy seed
    random.seed(local_seed) # Set the Python seed

    # torch._dynamo.config.log_level = logging.DEBUG
    torch._dynamo.config.verbose = True
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.debug = True

    class SelfAttnLikeModule(torch.nn.Module):
        def __init__(
            self,
            input_dim,
            transpose_for_score = False,
            num_attention_heads = None,
            attention_head_size = None,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.q_proj = nn.Linear(input_dim, input_dim, bias=False)
            self.k_proj = nn.Linear(input_dim, input_dim, bias=False)
            self.v_proj = nn.Linear(input_dim, input_dim, bias=False)
            self.softmax = nn.Softmax(dim=-1)
            self.transpose_for_score = transpose_for_score
            if self.transpose_for_score:
                assert num_attention_heads is not None
                assert attention_head_size is not None
                self.num_attention_heads = num_attention_heads
                self.attention_head_size = attention_head_size

        def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(new_x_shape)
            return x.permute(0, 2, 1, 3)

        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            if self.transpose_for_score:
                q = self.transpose_for_scores(q)
                k = self.transpose_for_scores(k)
                v = self.transpose_for_scores(v)
            # breakpoint()
            scores = torch.matmul(q, k.transpose(-1, -2)) / (self.input_dim ** 0.5)
            attention = self.softmax(scores)
            weighted = torch.matmul(attention, v)
            return weighted

    with torch.no_grad():
        m = SelfAttnLikeModule(
            input_dim=64*16,
            transpose_for_score=True,
            num_attention_heads=16,
            attention_head_size=64,
        ).eval()

        example_inputs = (torch.randn(2, 384, 1024),)

        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )


        annotate_matmul = True
        if annotate_matmul:
            quantizer.add_extra_quantizable_op(torch.ops.aten.matmul.default)

        from torch._export import capture_pre_autograd_graph

        m = capture_pre_autograd_graph(
            m,
            example_inputs,
        )

        prepared_model = prepare_pt2e(m, quantizer)
        prepared_model(*example_inputs)
        convert_model = convert_pt2e(prepared_model)

        with torch.autocast(device_type="cpu", enabled=True):
            compiled_model = torch.compile(convert_model)
            compiled_model(*example_inputs)
            compiled_model(*example_inputs)

    
        print("Finish the test", flush=True)


if __name__ == "__main__":
    test_sdpa_pattern()
