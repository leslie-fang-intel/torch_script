import torch
import numpy as np
import random

local_seed = 2022
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv2 = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv3 = torch.nn.Conv2d(128, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        # x1 = self.conv(x)
        # Here we need conv2 to ensure the first conv's output is int8
        # return self.conv3(x1)
        return self.conv(x)

def test_pytorch_module():
    from torch.ao.quantization import QConfigMapping
    import torch.quantization.quantize_fx as quantize_fx
    with torch.no_grad():
        batch_size = 1
        model = SimpleNet().eval()
        x = torch.rand(batch_size, 64, 3, 3)
        example_inputs = (x, )
        res_ref = model(x)
        torch.backends.quantized.engine = 'onednn'
        qconfig_mapping = QConfigMapping().set_global(torch.quantization.get_default_qconfig('onednn'))
        backend_config = torch.ao.quantization.backend_config.executorch.get_executorch_backend_config()
        model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs, backend_config=backend_config)

        # print("model_prepared is: {}".format(model_prepared), flush=True)
        # exit(-1)
        # calibrate (not shown)
        # model_prepared(x)
        for i in range(1):
            images = torch.rand(batch_size, 64, 3, 3)
            model_prepared(images)
        # quantize
        model_quantized = quantize_fx.convert_fx(model_prepared, backend_config=backend_config)

        model_quantized = torch.jit.trace(model_quantized, x)
        model_quantized = torch.jit.freeze(model_quantized.eval())

        for i in range(3):
            model_quantized(x)
        print(model_quantized.graph_for(x), flush=True)

        print("Final run")
        res = model_quantized(x)

        # print("res is:{}".format(res))
        
        print(torch.allclose(res_ref, res, rtol=0.08, atol=0.01))
        assert torch.allclose(res_ref, res, rtol=0.08, atol=0.01)

if __name__ == "__main__":
    test_pytorch_module()
