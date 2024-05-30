import torch
import torch._inductor.config

# torch._inductor.config.force_disable_caches = True  # doesn't seem to help

print(torch.__version__)

def func(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 2.0 ** (-127 + 3)

x = torch.tensor(0.1875)
print("Eager (before):", func(x))
print("Compile:", torch.compile(func)(x))
print("Eager (after):", func(x))
