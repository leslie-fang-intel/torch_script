import torch
import torchvision.models as models
from torch.fx.passes import graph_drawer

if __name__ == "__main__":


    model = models.__dict__["resnet50"](pretrained=True)
    model = model.to(memory_format=torch.channels_last).eval()

    fx_graph_module = torch.fx.symbolic_trace(model)
    # import pdb;pdb.set_trace()
    g = graph_drawer.FxGraphDrawer(fx_graph_module, "resnet50")
    # with open("resnet50.svg", "w") as f:
    #     f.write(g.get_dot_graph().create_svg())
    g.get_dot_graph().write_png('resnet50.png')
