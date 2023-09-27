import torch
import random
import numpy as np

random.seed(2023)
torch.manual_seed(2023)
np.random.seed(2023)

torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.debug_log = True
torch._inductor.config.debug = True

@torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
def test_nonzero_size_factory_nobreak(device):
    def f(x, b):
        # Pass
        # return b.size(0)
        
        # Pass        
        # y = x.relu()
        # # print("y is: {}".format(y), flush=True)
        # return y.size(0)

        # Pass
        # y = torch.nonzero(b)
        # return y

        # Failed
        # y = torch.nonzero(b)
        # return y.size(0)
        
        # pass
        # size = x.size(0)
        # return x.new_zeros(size)

        # Failed
        y = torch.nonzero(b)
        size = y.size(0)
        return x.new_zeros(size)


        # return x.relu()

    opt_f = torch.compile(f, fullgraph=True, dynamic=True)
    x = torch.randn(5, device=device)
    b = torch.tensor([True, True, False, False, True], device=device)
    r = f(x, b)
    print("r is: {}".format(r), flush=True)
    
    opt_r = opt_f(x, b)

    print("opt_r is: {}".format(opt_r), flush=True)


    # x2 = torch.randn((5, 2), device=device)
    # opt_r = opt_f(x2, b)

    # print(torch.allclose(r, opt_r, atol=1e-6, rtol=1e-6), flush=True)

if __name__ == "__main__":
    test_nonzero_size_factory_nobreak("cpu")
