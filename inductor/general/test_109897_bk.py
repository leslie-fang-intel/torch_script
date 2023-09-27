import torch

torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.debug_log = True
torch._inductor.config.debug = True

@torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
def test_nonzero_size_factory_nobreak(device):
    def f(x, b):
        y = torch.nonzero(b)
        return x.new_zeros(y.size(0))

    opt_f = torch.compile(f, fullgraph=True)
    x = torch.randn(5, device=device)
    b = torch.tensor([True, True, False, False, True], device=device)
    r = f(x, b)
    opt_r = opt_f(x, b)
    # self.assertEqual(r, opt_r)

if __name__ == "__main__":
    test_nonzero_size_factory_nobreak("cpu")
