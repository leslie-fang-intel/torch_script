import torch
import torch._inductor.config

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch._inductor import config as inductor_config
torch.manual_seed(122)
import functools
from torch._inductor import config as inductor_config
inductor_config.profiler_mark_wrapper_call = True
inductor_config.cpp.enable_kernel_profile = True
import functools
import random
import string
import unittest
from collections import namedtuple
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple, Union
from unittest import expectedFailure, skip, skipUnless
from unittest.mock import patch

import torch
from torch._dynamo.testing import CompileCounterWithBackend, normalize_gm
from torch._inductor import metrics
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.nn.attention.experimental._paged_attention import PagedAttention
from torch.nn.attention.flex_attention import (
    _create_empty_block_mask,
    _DEFAULT_SPARSE_BLOCK_SIZE,
    _identity,
    _mask_mod_signature,
    _score_mod_signature,
    and_masks,
    BlockMask,
    create_block_mask,
    flex_attention,
    noop_mask,
    or_masks,
)

index = torch.ops.aten.index
Tensor = torch.Tensor
B = 120 # 4
H = 32
S = 1152 # 2048
# S = 1
D = 128
def query_key_value_clones(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: torch.dtype = None,
):
    """Clones the query, key, and value tensors and moves them to the specified dtype."""
    if dtype is None:
        dtype = query.dtype
    query_ref = query.clone().detach().to(dtype).requires_grad_(query.requires_grad)
    key_ref = key.clone().detach().to(dtype).requires_grad_(key.requires_grad)
    value_ref = value.clone().detach().to(dtype).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref

def batch_reserve(paged_attention: PagedAttention, target_seq_len: Tensor):
    (B,) = target_seq_len.shape
    for b in range(B):
        paged_attention.reserve(
            torch.tensor(b),
            target_seq_len[b],
        )

def preprocess_paged_attention(
    score_mod: Optional[Callable],
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_mask,
    dtype: torch.dtype = torch.bfloat16,
    page_size: int = 128,
) -> Tuple[Tensor, Tensor, BlockMask, _score_mod_signature]:

    assert block_mask is not None, "Must provide block_mask"
    Q_B, Q_H, Q_S, _ = q.shape
    KV_B, KV_H, KV_S, QK_D = k.shape
    _, _, _, V_D = v.shape

    # test with different batch size
    max_batch_size = max(Q_B, KV_B) + 3

    n_pages = (KV_S + page_size - 1) // page_size * max_batch_size

    # allocate cache
    MAX_CACHED_SEQ_LEN = n_pages * page_size

    k_cache = torch.zeros(
        1,
        KV_H,
        MAX_CACHED_SEQ_LEN,
        QK_D,
        device="cpu",
        dtype=dtype,
    )
    v_cache = torch.zeros(
        1,
        KV_H,
        MAX_CACHED_SEQ_LEN,
        V_D,
        device="cpu",
        dtype=dtype,
    )

    # For testing purposes, we randomly initialize the page table, which maps
    # (batch_idx, logical_block_idx) to physical_block_idx. Specifically, PagedAttention
    # maintains a stack empty_pages of unused physical_block_idx. The `batch_reserve`
    # function grabs physical_block_idx from the top of empty_pages until there are enough
    # pages for each batch index (i.e., num pages for batch_idx >= target_seq_len[batch_idx]).
    # For example, at the first batch_reserve call, physical block indices (1,...,KV_S//4)
    # are allocated to batch index 0, and physical block indices
    # (KV_S//4+1, ..., KV_S//4 + KV_S//2) are allocated to batch index 1, etc.
    # Thus, kv tensors of batch index 1 will be scattered in the kv cache, simulating
    # a real use case of paged attention.
    paged_attention = PagedAttention(n_pages, page_size, max_batch_size, device="cpu")

    batch_reserve(
        paged_attention,
        torch.tensor([KV_S // 4, KV_S // 2, KV_S // 4, KV_S // 3], device="cpu"),
    )

    batch_reserve(
        paged_attention,
        torch.tensor([KV_S // 4, KV_S // 2, KV_S // 2, KV_S // 2], device="cpu"),
    )

    batch_reserve(
        paged_attention,
        torch.tensor([KV_S // 2, KV_S, KV_S // 2, KV_S], device="cpu"),
    )

    batch_reserve(
        paged_attention, torch.tensor([KV_S, KV_S, KV_S, KV_S], device="cpu")
    )

    # update cache with k and v
    input_pos = torch.arange(KV_S, device="cpu", dtype=torch.int32)
    batch_idx = torch.arange(KV_B, device="cpu", dtype=torch.int32)
    import time
    from torch._inductor import config as inductor_config
    inductor_config.profiler_mark_wrapper_call = True
    inductor_config.cpp.enable_kernel_profile = True

    warm_up_step = 20
    run_step = 100
    for _ in range(warm_up_step):
        paged_attention.assign(batch_idx, input_pos, k, v, k_cache, v_cache)

    s = time.time()
    for _ in range(run_step):
        paged_attention.assign(batch_idx, input_pos, k, v, k_cache, v_cache)
    e_2 = (time.time() - s)
    print("assign time:", e_2/run_step)

    def trace_handler(prof):
        print(prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=-1))
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=1),
            on_trace_ready=trace_handler
            ) as prof:
        s = time.time()
        paged_attention.assign(batch_idx, input_pos, k, v, k_cache, v_cache)
        e_2 = (time.time() - s)
        prof.step()
    print("assign time:", e_2)

    exit()

    return k_cache, v_cache, converted_block_mask, converted_score_mod

def run_paged_attention(
    score_mod: Optional[Callable],
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dtype: torch.dtype = torch.bfloat16,
    block_mask: Optional[BlockMask] = None,
) -> Tuple[Tensor, Tensor]:
    B, Q_H, Q_S, KV_H, KV_S = (
        q.shape[0],
        q.shape[1],
        q.shape[2],
        k.shape[1],
        k.shape[2],
    )
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    if block_mask is None:
        block_mask = create_block_mask(noop_mask, B, 1, Q_S, KV_S)

    (
        k_cache,
        v_cache,
        converted_block_mask,
        converted_score_mod,
    ) = preprocess_paged_attention(
        score_mod, q, k, v, block_mask, dtype, block_mask.BLOCK_SIZE[1]
    )


def create_attention(score_mod, block_mask, enable_gqa=False):
    return functools.partial(
        flex_attention,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
    )

def run_test_with_paged_attention(
    score_mod: Optional[Callable] = _identity,
    dtype: torch.dtype = torch.bfloat16,
    Q_B: int = B,
    Q_H: int = H,
    Q_S: int = 1,
    QK_D: int = D,
    KV_B: int = B,
    KV_H: int = H,
    KV_S: int = S,
    V_D: int = D,
    block_mask: Optional[BlockMask] = None,
):
    assert Q_H % KV_H == 0

    q = torch.randn(
        (Q_B, Q_H, Q_S, QK_D), dtype=dtype, device="cpu", requires_grad=False
    )
    k = torch.randn(
        (KV_B, KV_H, KV_S, QK_D), dtype=dtype, device="cpu", requires_grad=False
    )
    v = torch.randn(
        (KV_B, KV_H, KV_S, V_D), dtype=dtype, device="cpu", requires_grad=False
    )
    q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
    q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

    if block_mask is None:
        block_mask = create_block_mask(noop_mask, Q_B, Q_H, Q_S, KV_S,device="cpu")

    compiled_out = run_paged_attention(
        score_mod, q, k, v, dtype, block_mask
    )

run_test_with_paged_attention()
