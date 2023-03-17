import torch
import torch.fx as fx
import torch.ao.quantization.fx._decomposed
from torch._inductor.compile_fx import compile_fx

class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.scale = torch.tensor(0.003, dtype=torch.float)
        self.zp = torch.tensor(0, dtype=torch.int8)

    def forward(self, x):
        quant_per_tensor = torch.ops.quantized_decomposed.quantize_per_tensor(x, self.scale, self.zp, 0, 127, torch.uint8)
        dequant_tensor = torch.ops.quantized_decomposed.dequantize_per_tensor(quant_per_tensor, self.scale, self.zp, 0, 127, torch.uint8)
        return dequant_tensor + x
        #return quant_per_tensor

module = fx.symbolic_trace(M().eval()).eval()

# with torch.no_grad():
print("model before is: {}".format(module), flush=True)
example_inputs = (torch.randn(2, 3, 4, 4),)
optimized_model = compile_fx(module, example_inputs)
# Inductor first run
inductor_res = optimized_model(*example_inputs)
# Inductor second run
inductor_res = optimized_model(*example_inputs)

# # Insert quant and dequant node into graph
# graph = module.graph
# for node in module.graph.nodes:
#     print("node is: {}".format(node.op), flush=True)
#     if node.op == 'placeholder':
#         with module.graph.inserting_after(node):
#             scale = torch.tensor(0.003, dtype=torch.float)
#             zp = torch.tensor(0, dtype=torch.int8)
#             setattr(module, "scale", scale)
#             setattr(module, "zp", zp)
#             scale_node = module.graph.get_attr("scale")
#             zp_node = module.graph.get_attr("zp")
#         with module.graph.inserting_after(zp_node):
#             # Insert quant node
#             args = (node, scale_node, zp, 0, 127, torch.uint8)
#             quant_node = module.graph.call_function(
#                 torch.ops.quantized_decomposed.quantize_per_tensor, args=args
#             )
#             # Insert dequant node
#             print("print here", flush=True)
#             #node.replace_all_uses_with(quant_node)
# module.graph.lint()
# module.recompile()
# print("model after is: {}".format(module), flush=True)


