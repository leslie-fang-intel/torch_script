# Execution Command:
# TORCHINDUCTOR_FREEZING=1 python example_x86inductorquantizer_pytorch_2_1.py
import torch
import torchvision.models as models
import torch._dynamo as torchdynamo
import copy
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import time
from torch._export import capture_pre_autograd_graph, dynamic_dim
import torch.nn as nn

# torch._dynamo.config.verbose = True
# torch._inductor.config.trace.enabled = True
# torch._inductor.config.trace.debug_log = True
# torch._inductor.config.debug = True
torch._inductor.config.freezing = True


class Conv2dUnaryModule(torch.nn.Module):
    def __init__(self, post_op, use_bias: bool = False, with_bn=False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, (2, 2), stride=(1, 1), padding=(1, 1), bias=use_bias)
        self.post_op1 = copy.deepcopy(post_op)
        self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(24, 58, (2, 2), stride=(1, 1), padding=(1, 1), bias=use_bias)
        self.conv3= nn.Conv2d(24, 58, (2, 2), stride=(1, 1), padding=(1, 1), bias=use_bias)
        self.conv4= nn.Conv2d(58, 58, (3, 3), stride=(1, 1), padding=(1, 1), bias=use_bias)

    def forward(self, x):
        x = self.post_op1(self.conv1(x))
        # x = self.maxpool(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        tmp = torch.cat((x2, x3), 1)

        # channel_shuffle
        tmp = tmp.view(128, 2, 58, 226, 226)
        tmp = torch.transpose(tmp, 1, 2).contiguous()
        tmp = tmp.view(128, 116, 226, 226)
        # tmp = tmp.view(128, 116, 226, 226).contiguous(memory_format=torch.channels_last)

        # Chunk 
        tmp1, tmp2 = tmp.chunk(2, dim=1)

        tmp2 = self.conv4(tmp2)

        tmp = torch.cat((tmp1, tmp2), dim=1)
        return tmp

def run_rn50():
    batch_size = 128
    # model = models.__dict__["resnet50"](pretrained=True).eval()
    model = Conv2dUnaryModule(torch.nn.ReLU()).eval()
    
    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)

    # return model(x)

    example_inputs = (x,)
    with torch.no_grad():
        # Generate the FX Module
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
        
        print("prepared_model is: {}".format(prepared_model), flush=True)
        # from torch.fx.passes.graph_drawer import FxGraphDrawer
        # g = FxGraphDrawer(prepared_model, "resnet50")
        # g.get_dot_graph().write_svg("/home/lesliefang/pytorch_1_7_1/quantization/prepare_model.svg")


        converted_model = convert_pt2e(prepared_model)
        torch.ao.quantization.move_exported_model_to_eval(converted_model)

        print("converted_model is: {}".format(converted_model), flush=True)

        # from torch.fx.passes.graph_drawer import FxGraphDrawer
        # g = FxGraphDrawer(converted_model, "resnet50")
        # g.get_dot_graph().write_svg("./converted_model_rn50.svg")

        optimized_model = torch.compile(converted_model)
        print("start first run", flush=True)
        _ = optimized_model(*example_inputs)
        print("start second run", flush=True)
        _ = optimized_model(*example_inputs)
        print("---- finised ----", flush=True)
            # # Benchmark
            # warm_up_step = 100
            # measurement_step = 100
            # total_time = 0.0
            # for _ in range(warm_up_step):
            #     _ = optimized_model(*example_inputs)
            # for i in range(measurement_step):
            #     start_time = time.time()
            #     _ = optimized_model(*example_inputs)
            #     total_time += (time.time() - start_time)
            #     if (i+1) % 10 == 0:
            #         print("steps:{0}; throughput is: {1} fps".format((i+1), batch_size/(total_time/(i+1))), flush=True)

if __name__ == "__main__":
    run_rn50()