import torch
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig, HistogramObserver
import numpy as np
import random

local_seed = 2022
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed
import torch.nn as nn
CALI=True
import utils_vis

class SimpleNet(torch.nn.Module):
    def __init__(self,):
        super(SimpleNet, self).__init__()

    def forward(self, x, y):
        x = x.view(2, 2)
        # x = torch.matmul(x, x.transpose(0, 1))

        x = x / 1.0
        y = y.view(2, 2)
        y = (1.0 - y) * torch.finfo(torch.float32).min

        
        # y = y.view(2, 2)
        # x = torch.matmul(y, y)
        # return x
        
        # x = x + y
        x = torch.add(x, y)

        # x = torch.add(torch.tensor(x), torch.tensor(y))
        
        y = nn.functional.softmax(x)
        return y


if __name__ == "__main__":

    batch_size = 1

    model = SimpleNet()
    x = torch.rand(2, 2)

    x2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    res_ref = model(x, x2)

    print(res_ref.size(), flush=True)

    # exit(0)

    # qconfig = QConfig(
    #         activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
    #         weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))

    qconfig = QConfig(
            activation=HistogramObserver.with_args(reduce_range=False),
            weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))

    # qconfig = ipex.quantization.default_static_qconfig
    
    # qconfig = QConfig(
    #         activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
    #         weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)) 
    prepared_model = ipex.quantization.prepare(model, qconfig, (x, x2), inplace=True)

    print("Prepared model is:{}".format(prepared_model), flush=True)
    #exit(-1)
    if CALI:
        with  torch.cpu.amp.autocast(dtype=torch.bfloat16, enabled=False), torch.no_grad():
            #for i, (images, target) in enumerate(val_loader):
            for i in range(3):

                print("start calbri step: {}".format(i), flush=True)
                images = torch.rand(2, 2)
                
                # images2 = torch.rand(2, 2)

                images2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
                # images = images.contiguous(memory_format=torch.channels_last)
                prepared_model(images, images2)
                

            # prepared_model.save_qconf_summary("./cali.json")

            #prepared_model.load_qconf_summary(qconf_summary="./cali.json")
            model = ipex.quantization.convert(prepared_model)

            print("Converted model is:{}".format(model), flush=True)

            model = torch.jit.trace(model, (x, x2))
            model = torch.jit.freeze(model.eval())
            print("Graph before enter into LLGA")
            print(model.graph_for(x, x2), flush=True)
            
            for i in range(3):
                model(x, x2)
            print(model.graph_for(x, x2), flush=True)
            fwd_graph = model.graph_for(x, x2)
            print("Finish Print the graph", flush=True)
            model(x, x2)
            utils_vis.draw(fwd_graph).render("test")

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
            
