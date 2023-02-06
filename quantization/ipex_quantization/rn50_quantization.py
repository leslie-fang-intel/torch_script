import torch
import intel_extension_for_pytorch as ipex
import torchvision.models as models
from intel_extension_for_pytorch.quantization import prepare, convert
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
import numpy as np
import random

local_seed = 2022
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

CALI=False

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv2 = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv3 = torch.nn.Conv2d(128, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x, x2):
        # return self.conv(x)
        # print(self.conv(x).size())
        # return self.relu(torch.add(x2, self.conv(x)))
        
        #x2 = self.conv2(x)
        # conv add relu cases
        y1 = self.relu(self.conv(x) + self.conv2(x))

        return self.conv3(y1)
        # # conv add case
        # res = self.conv(x) + self.conv2(x)
        # # print(res.size())
        # return self.conv3(res)

if __name__ == "__main__":

    batch_size = 1
    # model = models.__dict__["resnet50"](pretrained=True).eval()
    # x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    model = SimpleNet().eval()
    x = torch.rand(batch_size, 64, 3, 3)
    x2 = torch.rand(batch_size, 128, 2, 2)
    res_ref = model(x, x2)
    
    
    # exit(-1)

    qconfig = QConfig(
            activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
            weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    # qconfig = QConfig(
    #         activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
    #         weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)) 
    prepared_model = ipex.quantization.prepare(model, qconfig, (x, x2), inplace=True)
    if CALI:
        with torch.no_grad():
            #for i, (images, target) in enumerate(val_loader):
            for i in range(1):
                # images = torch.randn(batch_size, 3, 224, 224)
                images = torch.rand(batch_size, 64, 3, 3)
                images2 = torch.rand(batch_size, 128, 2, 2)
                # images = images.contiguous(memory_format=torch.channels_last)
                prepared_model(images, images2)

            prepared_model.save_qconf_summary("./cali.json")

            #prepared_model.load_qconf_summary(qconf_summary="./cali.json")
            model = ipex.quantization.convert(prepared_model)

            model = torch.jit.trace(model, (x, x2))
            model = torch.jit.freeze(model.eval())
            # print("Graph before enter into LLGA")
            # print(model.graph_for(x, x2), flush=True)
            
            for i in range(3):
                model(x, x2)
            print(model.graph_for(x, x2), flush=True)
            print("Finish Print the graph", flush=True)
            model(x, x2)

    else:
        prepared_model.load_qconf_summary(qconf_summary="./cali.json")
        model = ipex.quantization.convert(prepared_model)
        with torch.no_grad():
            model = torch.jit.trace(model, (x, x2))
            model = torch.jit.freeze(model.eval())
            # print("Graph before enter into LLGA")
            # print(model.graph_for(x, x2), flush=True)
            
            for i in range(3):
                model(x, x2)
            print(model.graph_for(x, x2), flush=True)
            print("Finish Print the graph", flush=True)
            res = model(x, x2)
        
            print(torch.allclose(res_ref, res, rtol=0.08, atol=0.01))
            assert torch.allclose(res_ref, res, rtol=0.08, atol=0.01)
            
