import torch

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
        if kwargs is None:
            kwargs = {}
        print(type(types), flush=True)
        print(types.__len__(), flush=True)
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, ScalarTensor))
            for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

import functools
def implements(torch_function):
    """Register a torch function override for ScalarTensor"""
    print("---- outside in implements ----", flush=True)
    @functools.wraps(torch_function)
    def decorator(func):
        # import pdb;pdb.set_trace()
        print("---- add torch_function:{0}, func:{1} into HANDLED_FUNCTIONS ----".format(torch_function, func), flush=True)
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    print("---- return from implements ----", flush=True)
    return decorator

print("---- point1 -----", flush=True)
@implements(torch.mean)
def mean(input):
    return float(input._value) / input._N
print("---- point2 -----", flush=True)

if __name__ == "__main__":
    print("---- start the main program ----", flush=True)
    d = ScalarTensor(5, 2)
    res = torch.mean(d)
    print("res is: {}".format(res), flush=True)
