import torch
from itertools import product

# def test1():
#     a = torch.randn(2, 512, 65536)
#     select = torch.ops.aten.select.int(a, 0, 0)
#     view_3 = torch.ops.aten.reshape.default(select, [512, 32, -1, 128])
#     # reset the value of view_3
#     view_3 = torch.randn(512, 32, 16, 128)
#     print("view_3 after reset is: {}".format(view_3), flush=True)
#     print("original is: {}".format(
#         torch.ops.aten.reshape.default(torch.ops.aten.select.int(a, 0, 0), [512, 32, -1, 128])
#     ), flush=True)
#     print(view_3 == torch.ops.aten.reshape.default(torch.ops.aten.select.int(a, 0, 0), [512, 32, -1, 128]), flush=True)


# def test2():

#     a = torch.randn(2, 512, 65536)

#     def func(kv_cache):
#         num_blocks = kv_cache.shape[1]
#         key_cache = kv_cache[0]
#         key_cache = key_cache.view(num_blocks, 32, -1, 128)
#         value_cache = kv_cache[1]
#         value_cache = value_cache.view(num_blocks, 32, -1, 128)
#         return key_cache, value_cache 

#     cfn = torch.compile(func)

#     cfn(a)


import random
import intel_extension_for_pytorch as ipex
from typing import Tuple, List

def test(key, value, kv_cache, slot_mapping, num_blocks, num_head, head_size):

    key_cache = kv_cache[0]
    key_cache = key_cache.view(num_blocks, num_head, -1, head_size)
    value_cache = kv_cache[1]
    value_cache = value_cache.view(num_blocks, num_head, -1, head_size)

    torch.ops.torch_ipex.reshape_and_cache(
        key, value, key_cache, value_cache, slot_mapping
    )
    return

ctest = torch.compile(test, fullgraph=True)

def create_kv_caches(
    num_blocks: int,
    block_size: int,
    num_layer: int,
    num_head: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)

    scale = head_size**-0.5
    key_cache_shape = (num_blocks, num_head, block_size, head_size)
    key_caches = []
    for _ in range(num_layer):
        key_cache = torch.empty(size=key_cache_shape, dtype=dtype)
        key_cache.uniform_(-scale, scale)
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_head, block_size, head_size)
    value_caches = []
    for _ in range(num_layer):
        value_cache = torch.empty(size=value_cache_shape, dtype=dtype)
        value_cache.uniform_(-scale, scale)
        value_caches.append(value_cache)
    return key_caches, value_caches

def _test_reshape_and_cache_func(
    num_token: int,
    num_head: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)

    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_token)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int, device="cpu")

    qkv = torch.randn(num_token, 3, num_head, head_size, dtype=dtype, device="cpu")
    _, key, value = qkv.unbind(dim=1)
    # Create the KV caches.
    key_caches, value_caches = create_kv_caches(
        num_blocks, block_size, 1, num_head, head_size, dtype, seed
    )
    key_cache, value_cache = key_caches[0], value_caches[0]
    # Clone the KV caches.
    cloned_key_cache = key_cache.clone()
    cloned_value_cache = value_cache.clone()

    # Call the reshape_and_cache kernel.

    key_cache = key_cache.view(1, num_blocks, -1)
    value_cache = value_cache.view(1, num_blocks, -1)

    kv_caches = torch.stack((key_cache, value_cache), dim=0)

    ctest(key, value, kv_caches, slot_mapping, num_blocks, num_head, head_size)


    # Run the reference implementation.
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets = block_offsets.cpu().tolist()
    for i in range(num_token):
        for j in range(num_head):
            block_idx = block_indicies[i]
            block_offset = block_offsets[i]
            cloned_key_cache[block_idx, j, block_offset, :] = key[i][j]
            cloned_value_cache[block_idx, j, block_offset, :] = value[i][j]

def test3():
    num_blocks = 128  # Arbitrary values for testing
    num_tokens = [1024,]  # Arbitrary values for testing
    num_kv_heads = [8]  # Arbitrary values for testing
    head_sizes = [64,]
    block_sizes = [16,]
    dtypes = [torch.bfloat16,]
    seeds = [0]
    for (
        num_token,
        num_kv_head,
        head_size,
        block_size,
        dtype,
        seed,
    ) in product(
        num_tokens,
        num_kv_heads,
        head_sizes,
        block_sizes,
        dtypes,
        seeds,
    ):
        _test_reshape_and_cache_func(
            num_token, num_kv_head, head_size, block_size, num_blocks, dtype, seed
        )


if __name__ == "__main__":
    # test1()
    test3()