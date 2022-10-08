import torch

input_tensor = torch.rand(3, 4)
input_tensor2 = torch.rand(3, 4)

result = torch.add(input_tensor, input_tensor2)

HANDLED_FUNCTIONS = {}
class ScalarTensor(object):
    def __init__(self, N, value):
        self._N = N
        self._value = value

    def __repr__(self):
        return "DiagonalTensor(N={}, value={})".format(self._N, self._value)

    def tensor(self):
        return self._value * torch.eye(self._N)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        #import pdb;pdb.set_trace()
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, ScalarTensor))
            for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

import functools
def implements(torch_function):
    """Register a torch function override for ScalarTensor"""
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator

def ensure_tensor(data):
    if isinstance(data, ScalarTensor):
        return data.tensor()
    return torch.as_tensor(data)

@implements(torch.add)
def add(input, other):
   try:
       if input._N == other._N:
           return ScalarTensor(input._N, input._value + other._value)
       else:
           raise ValueError("Shape mismatch!")
   except AttributeError:
       return torch.add(ensure_tensor(input), ensure_tensor(other))

d = ScalarTensor(5, 2)
#print(d.tensor())

d2 = ScalarTensor(5, 1)
result = torch.add(d, d2)
print(result)
