import torch
import copy
from torch.ao.quantization._quantize_pt2e import (
    prepare_pt2e_quantizer,
    convert_pt2e
)
import torch._dynamo as torchdynamo
import intel_extension_for_pytorch as ipex

class Mod(torch.nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.maxpool = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = torch.nn.ReLU()

        self.conv4 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.conv(x)
        res1 = self.relu(self.conv2(x1) + self.conv3(x1))
        res2 = self.relu2(self.conv4(res1) + res1)
        return res2

def test_ipex_pt2e_ptq_static_quant():
    import torchvision
    example_inputs = (torch.randn(1, 3, 224, 224),)
    m = torchvision.models.resnet50().eval()

    # example_inputs = (torch.randn(1, 3, 16, 16),)
    # m = Mod().eval()

    r = m(*example_inputs)
    print(r.size())

    original_model = copy.deepcopy(m)

    with torch.no_grad():

        # program capture
        export_model, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        print("export_model is: {}".format(export_model), flush=True)

        quantizer = ipex.quantization.IPEXQuantizer()
        quantizer.set_global(ipex.quantization.get_default_ipex_quantization_config())
        prepared_model = prepare_pt2e_quantizer(export_model, quantizer)

        print("prepared_model is: {}".format(prepared_model), flush=True)

        # Calibration
        for i in range(3):
            prepared_model(*example_inputs)

        from torch.fx.passes.graph_drawer import FxGraphDrawer
        g = FxGraphDrawer(prepared_model, "resnet50")
        g.get_dot_graph().write_svg("./prepare_ipex_rn50_staic_quant.svg")

        quantized_model = convert_pt2e(prepared_model)
        print("quantized_model is: {}".format(quantized_model), flush=True)


        with torch.no_grad():
            model_ipex = torch.jit.trace(quantized_model, example_inputs).eval()
            model_ipex = torch.jit.freeze(model_ipex)
            y = model_ipex(*example_inputs)
            y = model_ipex(*example_inputs)
            print("model_ipex.graph_for(*example_inputs) is: {}".format(model_ipex.graph_for(*example_inputs)), flush=True)
        
        profiler = True
        if profiler:
            with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler")) as prof:
                _ = model_ipex(*example_inputs)
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        print("---- Finish the testing ----", flush=True)

if __name__ == "__main__":
    test_ipex_pt2e_ptq_static_quant()
