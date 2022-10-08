import torch
import torch.nn as nn
import torch.fx.experimental.optimization as optimization

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(torch.nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        torch.nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=(1, 1), bias=False))

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.layer1(y)
        #print("y.size: {}".format(y.size), flush=True)
        return y

class LayerAffinityModule(torch.nn.Module):
    def __init__(self, original_module):
        super(LayerAffinityModule, self).__init__()
        self.original_module = original_module
        print("---------------finish create LayerAffinityModule--------", flush=True)
        print("---------------self.original_module.size()--------", flush=True)
        self.children_modules = []
        for name, m in self.original_module.named_children():
            print(name, '->', m, flush=True)
            self.children_modules.append(m)
        # for m in self.original_module.children():
        #     self.children_modules.append(m)

    def forward(self, *args, **kwargs):
        # print(self.children_modules.__len__())
        # for m in self.children_modules:
        #     print(type(m), flush=True)
        # return self.original_module(*args, **kwargs)
        for idx in range(self.children_modules.__len__()):
            print("self.children_modules[{0}]: is:{1}".format(idx, self.children_modules[idx]))
            if idx == 0:
                res = self.children_modules[0](*args, **kwargs)
            elif idx == 9:
                res = torch.flatten(res, 1)
                res = self.children_modules[idx](res)
            else:
                if type(res) is tuple:
                    res = self.children_modules[idx](*res)
                else:
                    res = self.children_modules[idx](res)
            print("------type(res) is:{}".format(type(res)))
            print("------res.size() is:{}".format(res.size()))
        return res

if __name__ == "__main__":
    model = SimpleNet().eval()
    x = torch.rand(64, 64, 3, 3)

    print("----model1----------")
    model(x)

    print("----model2----------")
    model2 = LayerAffinityModule(model)
    model2(x)

    print("----FX_GRAPHModule----------")
    FX_GRAPHModule = optimization.fuse(model)
    FX_GRAPHModule(x)

    print("----FX_GRAPHModule Affinity----------")
    model4 = LayerAffinityModule(FX_GRAPHModule)
    model4(x)

    print("----------finish test-------------", flush=True)