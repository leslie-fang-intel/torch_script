import torch

if __name__ == "__main__":
    a = torch.tensor([1.2, 2.1], dtype=torch.bfloat16)
    b = torch.tensor([1.2, 2.1], dtype=torch.float32)

    def fn(a, b):
        # return a + b
        # a = a + 1
        return a.add_(b)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        # compile_fn = torch.compile(fn)

        # compile_fn(a, b)
        c = fn(a, b)
        print(a, flush=True)
        print(c, flush=True)


# c = a + b
# c2 = c.to(bfloat16)
# copy(c2, a)


# b = b.to(float32)
# c = a + b