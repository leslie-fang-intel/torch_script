import torch

input = torch.randn(2, 3, 224, 224)
scales = torch.ones((3,))
zero_points = torch.zeros((3,))

def _unsqueeze_multiple(x, dimensions):
    for dim in sorted(dimensions):
        x = torch.unsqueeze(x, dim)
    return x

axis = 1

broadcast_dims = list(range(0, axis)) + list(range(axis + 1, input.ndim))
print("broadcast_dims is: {}".format(broadcast_dims), flush=True)

unsqueeze_zero_points = _unsqueeze_multiple(zero_points, broadcast_dims)
print(unsqueeze_zero_points.size(), flush=True)

r = input - unsqueeze_zero_points

