import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq

class M(torch.nn.Module):
    def __init__(
        self,
        use_bias,
        post_op_algo,
    ):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 16, use_bias)
        self.unary_fn = torch.nn.GELU(approximate=post_op_algo)
        self.linear2 = torch.nn.Linear(16, 16, use_bias)

    def forward(self, x):
        temp = self.linear1(x)
        temp = self.unary_fn(temp)
        # temp = self.linear2(temp)
        return temp

def test_bf16():
    bias = True
    # "none", "tanh"
    mod = M(bias, "none").eval()

    v = torch.randn((3, 16))
    inputs = (v,)

    maybe_no_grad = torch.no_grad()
    with maybe_no_grad:
        enable_int8_mixed_bf16 = True

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enable_int8_mixed_bf16):
            compiler_mode = torch.compile(mod)
            _ = compiler_mode(*inputs)
            output = compiler_mode(*inputs)
   
def test_int8_bf16():
    bias = True
    # "none", "tanh"
    mod = M(bias, "none").eval()

    v = torch.randn((3, 16))
    inputs = (v,)

    maybe_no_grad = torch.no_grad()
    with maybe_no_grad:
        export_model = capture_pre_autograd_graph(
            mod,
            inputs,
        )
        quantizer = X86InductorQuantizer()
        quantizer.set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        prepare_model = prepare_pt2e(export_model, quantizer)
        prepare_model(*inputs)
        convert_model = convert_pt2e(prepare_model, fold_quantize=True)
        torch.ao.quantization.move_exported_model_to_eval(convert_model)

        print("convert_model is: {}".format(convert_model), flush=True)
        
        # from torch.fx.passes.graph_drawer import FxGraphDrawer
        # g = FxGraphDrawer(convert_model, "resnet50")
        # g.get_dot_graph().write_svg("/home/lesliefang/pytorch_1_7_1/quantization/torch_script/inductor/int8/test_int8_model/qlinear/prepare_model.svg")
        
        # enable_int8_mixed_bf16 = False
        enable_int8_mixed_bf16 = False

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enable_int8_mixed_bf16):
            compiler_mode = torch.compile(convert_model)
            _ = compiler_mode(*inputs)
            output = compiler_mode(*inputs)


if __name__ == "__main__":
    # test_bf16()
    test_int8_bf16()