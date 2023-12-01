import torch
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
import torch.nn.functional as F
import torch.ao.quantization.fx._decomposed

quantized_decomposed = torch.ops.quantized_decomposed

def test():
    def ut1_need_fx(x1, x2):
        # mimic first cat
        x1 = quantized_decomposed.dequantize_per_tensor.default(x1, 1.0, 0, 0, 255, torch.uint8)
        x2 = quantized_decomposed.dequantize_per_tensor.default(x2, 1.0, 0, 0, 255, torch.uint8)
        tmp = torch.cat((x1, x2), 1)
        tmp = quantized_decomposed.quantize_per_tensor.default(tmp, 1.0, 0, 0, 255, torch.uint8)
        tmp = quantized_decomposed.dequantize_per_tensor.default(tmp, 1.0, 0, 0, 255, torch.uint8)

        # channel_shuffle
        tmp = tmp.view(128, 2, 58, 224, 224)
        tmp = torch.transpose(tmp, 1, 2).contiguous()
        tmp = tmp.view(128, 116, 224, 224)

        # Chunk 
        tmp1, tmp2 = tmp.chunk(2, dim=1)


        # tmp2 = quantized_decomposed.quantize_per_tensor.default(tmp2, 1.0, 0, 0, 255, torch.uint8)
        # tmp2 = quantized_decomposed.dequantize_per_tensor.default(tmp2, 1.0, 0, 0, 255, torch.uint8)
        # tmp2 = torch.nn.functional.conv2d(tmp2, torch.randn(58, 58, 3, 3), padding=1)
        # tmp2 = quantized_decomposed.quantize_per_tensor.default(tmp2, 1.0, 0, 0, 255, torch.uint8)
        # tmp2 = quantized_decomposed.dequantize_per_tensor.default(tmp2, 1.0, 0, 0, 255, torch.uint8)

        tmp2 = quantized_decomposed.quantize_per_tensor.default(tmp2, 1.0, 0, 0, 255, torch.uint8)
        tmp2 = tmp2.contiguous(memory_format=torch.channels_last)
        tmp2 = F.relu(tmp2).contiguous(memory_format=torch.channels_last)
        tmp2 = quantized_decomposed.dequantize_per_tensor.default(tmp2, 1.0, 0, 0, 255, torch.uint8)

        tmp = torch.cat((tmp1, tmp2), dim=1)

        return tmp

    def ut2_need_fx(tmp):

        # Chunk 
        tmp1, tmp2 = tmp.chunk(2, dim=1)

        tmp2 = quantized_decomposed.quantize_per_tensor.default(tmp2, 1.0, 0, 0, 255, torch.uint8)
        tmp2 = tmp2.contiguous(memory_format=torch.channels_last)
        tmp2 = F.relu(tmp2).contiguous(memory_format=torch.channels_last)
        tmp2 = quantized_decomposed.dequantize_per_tensor.default(tmp2, 1.0, 0, 0, 255, torch.uint8)

        tmp = torch.cat((tmp1, tmp2), dim=1)

        return tmp

    def fn(x1, x2):
        # mimic first cat
        x1 = quantized_decomposed.dequantize_per_tensor.default(x1, 1.0, 0, 0, 255, torch.uint8)
        x2 = quantized_decomposed.dequantize_per_tensor.default(x2, 1.0, 0, 0, 255, torch.uint8)
        tmp = torch.cat((x1, x2), 1)
        tmp = quantized_decomposed.quantize_per_tensor.default(tmp, 1.0, 0, 0, 255, torch.uint8)
        tmp = quantized_decomposed.dequantize_per_tensor.default(tmp, 1.0, 0, 0, 255, torch.uint8)

        # channel_shuffle
        tmp = tmp.view(128, 2, 58, 224, 224)
        tmp = torch.transpose(tmp, 1, 2).contiguous()
        tmp = tmp.view(128, 116, 224, 224)

        # Chunk 
        tmp1, tmp2 = tmp.chunk(2, dim=1)


        # tmp2 = quantized_decomposed.quantize_per_tensor.default(tmp2, 1.0, 0, 0, 255, torch.uint8)
        # tmp2 = quantized_decomposed.dequantize_per_tensor.default(tmp2, 1.0, 0, 0, 255, torch.uint8)
        # tmp2 = torch.nn.functional.conv2d(tmp2, torch.randn(58, 58, 3, 3), padding=1)
        # tmp2 = quantized_decomposed.quantize_per_tensor.default(tmp2, 1.0, 0, 0, 255, torch.uint8)
        # tmp2 = quantized_decomposed.dequantize_per_tensor.default(tmp2, 1.0, 0, 0, 255, torch.uint8)

        tmp2 = quantized_decomposed.quantize_per_tensor.default(tmp2, 1.0, 0, 0, 255, torch.uint8)
        tmp2 = tmp2.contiguous(memory_format=torch.channels_last)
        tmp2 = F.relu(tmp2).contiguous(memory_format=torch.channels_last)
        tmp2 = quantized_decomposed.dequantize_per_tensor.default(tmp2, 1.0, 0, 0, 255, torch.uint8)

        tmp = torch.cat((tmp1, tmp2), dim=1)

        return tmp

    x1 = torch.clamp(
        torch.randn((128, 58, 224, 224), dtype=torch.float32) * 100, 0, 255
    ).to(torch.uint8).to(memory_format=torch.channels_last)

    x2 = torch.clamp(
        torch.randn((128, 58, 224, 224), dtype=torch.float32) * 100, 0, 255
    ).to(torch.uint8).to(memory_format=torch.channels_last)

    zero_point = 100
    scale = 0.01
    def compile_fx_wrapper(model_, example_inputs_):
        return compile_fx(model_, example_inputs_)

    def run(*ex, **kwargs):
        return fn(*ex, **kwargs)

    run = torch._dynamo.optimize(compile_fx_wrapper, nopython=True)(run)
    actual = run(x1, x2)

if __name__ == "__main__":


    test()