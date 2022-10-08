import torch
import copy

# Simple module for demonstration
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        #return self.linear(x + self.param).clamp(min=0.0, max=1.0)
        x2 = x + x
        return self.linear(x2)

module = MyModule()

from torch.fx import symbolic_trace
# Symbolic tracing frontend - captures the semantics of the module
original_graph_module : torch.fx.GraphModule = symbolic_trace(module)

# High-level intermediate representation (IR) - Graph representation
# print(original_graph_module.graph)
# print(original_graph_module.code)
original_graph_module.graph.print_tabular()
# print(original_graph_module.graph._len)

input_tensor = torch.rand(2, 4)
print(original_graph_module(input_tensor))

def test(model, x):
    traced_graph = torch.jit.trace(model, x).eval()
    traced_graph = torch.jit.freeze(traced_graph)

    # warm_up
    for _ in range(3):
        traced_graph(x)
    print(traced_graph.graph_for(x))
    return

# test(original_graph_module, input_tensor)

def graph_transformation(fx_graph_module):
    new_graph = copy.deepcopy(fx_graph_module.graph)
    for node in new_graph.nodes:
        print(node.name)
    new_graph.lint() # check the new graph is well formed
    return new_graph

def graph_transformation_MyModule(fx_graph_module):
    new_graph = copy.deepcopy(fx_graph_module.graph)
    # Step1: Insert the split node
    for node in new_graph.nodes:
        print(node.name)
        if node.name == "add":
            with new_graph.inserting_after(node):
                size = new_graph.call_method("size", (node, ), {'dim': 0}) # create size node
                with new_graph.inserting_after(size):
                    floordiv = new_graph.call_function(torch.div, (size, 2), {"rounding_mode": "floor"}) # get the mini bs of each node
                    with new_graph.inserting_after(floordiv):
                        split = new_graph.call_function(torch.split, (node, floordiv), {'dim': 0}) # split
                        with new_graph.inserting_after(split):
                            def get_item(tuple, idx):
                                return tuple[idx]
                            get_item0 = new_graph.call_function(get_item, (split, 0)) # item0
                            get_item1 = new_graph.call_function(get_item, (split, 1)) # item1
            #node.replace_all_uses_with()
        if node.name == "linear":
            # the split node already created before we visit linear
            node.args = (get_item0, )
            with new_graph.inserting_after(node):
                linear2 = new_graph.call_module("linear", (get_item1,))
                with new_graph.inserting_after(linear2):
                    cat = new_graph.call_function(torch.cat, ((node, linear2), ))
        if node.name == "output":
            node.args = (cat, )
    new_graph.lint() # check the new graph is well formed
    return new_graph

new_graph = graph_transformation_MyModule(original_graph_module)
new_graph_module = torch.fx.GraphModule(original_graph_module, new_graph)
test(new_graph_module, input_tensor)
print("------------new graph-------------")
print(new_graph_module.graph)
new_graph_module.graph.print_tabular()
new_graph_module.recompile()
#print(new_graph_module(input_tensor))
print("--------------start to test---------")
# import pdb; pdb.set_trace()
test(new_graph_module, input_tensor)
