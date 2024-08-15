import torch
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import time
from torch._export import capture_pre_autograd_graph
class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=1024, out_features=4096)

    def forward(self, attn_weights):
        attn_weights = self.linear(attn_weights)  
        return attn_weights

def test_pt2e_quant():
    with torch.no_grad():
        model = M().eval()
        x = torch.randn(2, 1024)
        example_inputs = (x,)
        exported_model = capture_pre_autograd_graph(
            model,
            example_inputs
        )

        print("exported_model is: {}".format(exported_model), flush=True)

        # Create X86InductorQuantizer
        quantizer = xiq.X86InductorQuantizer()
        quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
        # PT2E Quantization flow
        prepared_model = prepare_pt2e(exported_model, quantizer)
        # Calibration
        prepared_model(*example_inputs)

        converted_model = convert_pt2e(prepared_model)
        torch.ao.quantization.move_exported_model_to_eval(converted_model)
        print("converted_model is: {}".format(converted_model.linear_scale_0.size()), flush=True)

def torchao_GPTQ_int4():
    pass

if __name__ == "__main__":
    test_pt2e_quant()
