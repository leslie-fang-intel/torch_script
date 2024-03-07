import torch
import argparse
import torch.distributed as dist
from torch.distributed import ReduceOp
import torchvision
import torch.nn as nn
from torch._inductor import config as inductor_config
 
inductor_config.layout_optimization=True
# inductor_config.freezing=True
inductor_config.force_layout_optimization=True
# inductor_config.profile
# torch.utils.data.DataLoader
# model = torchvision.models.resnet50(pretrained=False)
torch.manual_seed(2023)
# model = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)#.to(memory_format=torch.channels_last)
# aas = [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
#                                nn.Conv2d(64 * 4, int(64 * 2), 3, stride=1, padding=1),
#                                nn.BatchNorm2d(64 * 2),
#                                nn.ReLU(True),]
mult = 2 ** 2
            # model_dec_al += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias),norm_layer(int(ngf * mult / 2)),nn.ReLU(True)]
model_dec_al = [
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
]
mult = 2
aas = model_dec_al
model = nn.Sequential(*aas)
 
 
model = model.to(device="cpu").eval()#.to(memory_format=torch.channels_last)
out_dec_al = torch.rand([1, 32, 32, 32], device="cpu").to(memory_format=torch.channels_last)
def compile_test(out_dec_al):
    with torch.no_grad():
        # x1 = torch.rand([1, 128, 256, 256], device="cpu").to(memory_format=torch.channels_last)
        # x2 = torch.rand([1, 128, 256, 256], device="cpu").to(memory_format=torch.channels_last)
       
        # out = model(out_dec_al)
        i = torch.arange(64, device=out_dec_al.device).to(dtype=out_dec_al.dtype)



        j = torch.arange(64, device=out_dec_al.device).to(dtype=out_dec_al.dtype)

        # print("i is: {}".format(i), flush=True)
        # print("j is: {}".format(j), flush=True)
 
        x_f32 = j * 0.5
        y_f32 = i * 0.5
        y_f32 = y_f32.unsqueeze(-1)
 
        x = x_f32.to(torch.int64)
        y = y_f32.to(torch.int64)
 
        # print("x is: {}".format(x.size()), flush=True)
        # print("y is: {}".format(y.size()), flush=True)
    
        v1 = torch._ops.ops.aten._unsafe_index(out_dec_al, [None, None, y, x])

        # print("v1 is: {}".format(v1.size()), flush=True)
        return v1 * 3
 
 
out = compile_test(out_dec_al)

# exit(-1)

my_fn = torch.compile(compile_test)
out = my_fn(out_dec_al)
out = my_fn(out_dec_al)