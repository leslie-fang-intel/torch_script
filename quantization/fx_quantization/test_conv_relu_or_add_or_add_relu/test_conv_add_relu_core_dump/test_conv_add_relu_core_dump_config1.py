import torch
import numpy as np
import random

local_seed = 2022
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

CALI = True

class SimpleNet(torch.nn.Module):
    def __init__(self, with_bias=False):
        super(SimpleNet, self).__init__()
        # Config1
        # self.conv1 = torch.nn.Conv2d(2, 2, (1, 2), stride=(1, 2), padding=(1, 1), bias=with_bias)
        # self.conv2 = torch.nn.Conv2d(2, 2, (1, 2), stride=(1, 2), padding=(1, 1), bias=with_bias)

        # Config2
        # self.conv1 = torch.nn.Conv2d(5, 5, (1, 7), stride=(2, 2), padding=(1, 1), bias=with_bias)
        # self.conv2 = torch.nn.Conv2d(5, 5, (1, 7), stride=(2, 2), padding=(1, 1), bias=with_bias)

        # Config3
        self.conv1 = torch.nn.Conv2d(360, 180, (1, 3), groups=90, stride=(1, 2), padding=(2, 2), dilation=2, bias=with_bias)
        self.conv2 = torch.nn.Conv2d(360, 180, (1, 3), groups=90, stride=(1, 2), padding=(2, 2), dilation=2, bias=with_bias)

        self.relu = torch.nn.ReLU()
    def forward(self, x, x2): 
        x2 = self.conv2(x2)       
        x1 = self.relu(torch.add(self.conv1(x), x2))
        return x1

def test_ipex_module():
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_pytorch.quantization import prepare, convert
    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
    batch_size = 1
    model = SimpleNet().eval()

    # Config1
    # x = torch.rand(batch_size, 2, 10, 7)
    # x2 = torch.rand(batch_size, 2, 10, 7)

    # Config2
    # x = torch.rand(batch_size, 5, 12, 12)
    # x2 = torch.rand(batch_size, 5, 12, 12)

    # Config3
    x = torch.rand(batch_size, 360, 12, 14)
    x2 = torch.rand(batch_size, 360, 12, 14)

    res_ref = model(x, x2)
    qconfig = QConfig(
            activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
            weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    prepared_model = ipex.quantization.prepare(model, qconfig, (x, x2), inplace=True)
    with torch.no_grad():
        for i in range(1):
            prepared_model(x, x2)
        prepared_model.save_qconf_summary("./ipex_cali.json")
        model = ipex.quantization.convert(prepared_model)
        model = torch.jit.trace(model, (x, x2))
        model = torch.jit.freeze(model.eval())
        for i in range(3):
            model(x, x2)
        print(model.graph_for(x, x2), flush=True)
        print("Finish Print the graph", flush=True)
        res_quantized = model(x, x2)
        import numpy as np
        np.testing.assert_array_almost_equal(res_ref.cpu().numpy(),
        res_quantized.cpu().numpy(), decimal=2)

if __name__ == "__main__":
    test_ipex_module()
