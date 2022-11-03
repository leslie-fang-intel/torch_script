import torch
from torch._decomp import decomposition_table
import torchvision.models as models
#import intel_extension_for_pytorch as ipex
from torch.fx import symbolic_trace
# import torch.fx as fx
import torch.nn.functional as F
import torch._dynamo

if __name__ == "__main__":
    #model = F.log_softmax
    model = models.__dict__["resnet50"]().to(memory_format=torch.channels_last).eval()
    # model = F.relu
    graph_module = symbolic_trace(model)
    print(graph_module.graph)

    x = torch.randn(28, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    export_graph, _ = torch._dynamo.export(
        model,
        (x),
        aten_graph=True,
        decomposition_table=decomposition_table
    )
    print(type(graph_module.graph))
    print(type(export_graph.graph))
