import torch
from torch._inductor import config as inductor_config
from torchao.quantization import quant_api
inductor_config.cpp_wrapper = True

with torch.no_grad():
    # For weight only int8
    quant_api.change_linear_weights_to_int8_woqtensors(user_model)

    # For weight only int4
    # quant_api.change_linear_weights_to_int4_woqtensors(user_model)

    with torch.no_grad(), torch.cpu.amp.autocast(
        enabled=True, dtype=torch.BFloat16
    ):
        user_model = torch.compile(user_model)
        user_model(example_input)
        user_model(example_input)
