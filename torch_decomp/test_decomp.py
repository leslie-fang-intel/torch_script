import torch
from torch._decomp import decomposition_table
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
import torchvision.models as models
from torch._dispatch.python import enable_python_dispatcher
#import intel_extension_for_pytorch as ipex
from torch.fx import symbolic_trace

class DecompCrossRefMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        with no_dispatch():
            return self._torch_dispatch(func, types, args, kwargs)

    def _torch_dispatch(self, func, types, args=(), kwargs=None):
        if func in decomposition_table:
            print(func)
            print("inside _torch_dispatch")

        result = None
        if func in decomposition_table:
            # Not every func in decomposition_table
            result = decomposition_table[func](*args, **kwargs)
            print(decomposition_table[func])
            print(result)
        else:
            result = func(*args, **kwargs)
        return result

if __name__ == "__main__":
    # print(type(decomposition_table))
    # for key in decomposition_table.keys():
    #     print("key:{0}, val:{1}".format(key, decomposition_table[key]))
    x = torch.randn(28, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    model = models.__dict__["resnet50"]().to(memory_format=torch.channels_last).eval()
    #model = ipex.optimize(model)
    with DecompCrossRefMode(), enable_python_dispatcher():
        graph_module = symbolic_trace(model)
    print(graph_module.graph)

