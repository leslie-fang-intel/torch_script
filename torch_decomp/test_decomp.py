import torch
from torch._decomp import decomposition_table
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
import torchvision.models as models
#import intel_extension_for_pytorch as ipex
from torch.fx import symbolic_trace
from torch.utils._python_dispatch import enable_torch_dispatch_mode
import torch.fx as fx
import torch.nn.functional as F

class DecompCrossRefMode(torch.Tensor):
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        with no_dispatch():
            return cls._torch_dispatch(func, types, args, kwargs)

    @classmethod
    def _torch_dispatch(cls, func, types, args=(), kwargs=None):
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
    
    #x = torch.randn(28, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    #model = models.__dict__["resnet50"]().to(memory_format=torch.channels_last).eval()
    
    #model = ipex.optimize(model)
    #with enable_torch_dispatch_mode(DecompCrossRefMode):
        #graph_module = symbolic_trace(model)
    #    model(x)
    # model = torch.ops.aten._log_softmax()
    model = F.log_softmax
    # model = F.relu
    graph_module = symbolic_trace(model)
    print(graph_module.graph)

    def relu_decomposition(x):
        return (x > 0) * x

    decomposition_rules = {}
    decomposition_rules[F.relu] = relu_decomposition
    
    for key in decomposition_table.keys():
        if "aten::_log_softmax" == key.name:
            print("-----")
            print(key)
            print(type(key))
            print(key.name)
            decomposition_rules[F.log_softmax] = decomposition_table[key]

    def transform(graph_module: torch.nn.Module) -> torch.nn.Module:
        graph = graph_module.graph
        new_graph = torch.fx.Graph()
        env = {}
        tracer = torch.fx.proxy.GraphAppendingTracer(new_graph)

        for node in graph.nodes:
            if node.op == 'call_function' and node.target in decomposition_rules:
                # By wrapping the arguments with proxies,
                # we can dispatch to the appropriate
                # decomposition rule and implicitly add it
                # to the Graph by symbolically tracing it.
                proxy_args = [
                    fx.Proxy(env[x.name], tracer) if isinstance(x, fx.Node) else x for x in node.args]
                # **TODO** Add the keyword parameter
                # for key in node.kwargs:
                #     x = node.kwargs[key]
                #     if isinstance(x, fx.Node):
                #         proxy_args.append(fx.Proxy(env[x.name], tracer))
                #     else:
                #         proxy_args.append(x)
                # **TODO** The parameters are not aligned between:
                # 1. torch.nn.functional.log_softmax: https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html
                # 2. _log_softmax inside: frameworks.ai.pytorch.private-cpu/torch/_decomp/decompositions.py
                proxy_args.append(0)
                proxy_args.append(False)
                    #proxy_args.append(node.kwargs[key])
                output_proxy = decomposition_rules[node.target](*proxy_args)

                # Operations on `Proxy` always yield new `Proxy`s, and the
                # return value of our decomposition rule is no exception.
                # We need to extract the underlying `Node` from the `Proxy`
                # to use it in subsequent iterations of this transform.
                new_node = output_proxy.node
                env[node.name] = new_node
            else:
                # Default case: we don't have a decomposition rule for this
                # node, so just copy the node over into the new graph.
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                env[node.name] = new_node
        return torch.fx.GraphModule(graph_module, new_graph)
    
    new_graph_module = transform(graph_module)
    print("new_graph_module")
    print(new_graph_module.graph)
