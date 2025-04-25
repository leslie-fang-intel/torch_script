import torch
import math
from itertools import product

@torch.library.impl("torch_cuda_extension::extended_attention", "cpu")
def extended_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, api_level=0):
    # q: [bs, num_head, seq_q, head_dim]
    # k/v: [bs, num_head, seq_kv, head_dim]
    if is_causal:
        assert attn_mask is None, "can't use is_causal and attn_mask at same time"
    if api_level == 0:
        q_seq_len = q.size(2)
        kv_seq_len = k.size(2)
        attn_bias = None
        if is_causal:
            attn_bias = torch.zeros((q_seq_len, kv_seq_len), dtype=q.dtype, device=q.device)
            attn_mask = torch.ones(q_seq_len, kv_seq_len, dtype=torch.bool, device=q.device).tril(diagonal=0)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        elif attn_mask is not None:
            attn_bias = torch.zeros((q_seq_len, kv_seq_len), dtype=q.dtype, device=q.device)
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias
        scale_factor = scale if scale is not None else 1 / math.sqrt(q.size(-1))
        attn_weight = torch.matmul(q, k.transpose(-2, -1))
        if attn_bias is not None:
            attn_weight += attn_bias
        attn_weight = attn_weight * scale_factor
        attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1)
        return torch.matmul(attn_weight, v)
    elif api_level == 1:
        output = torch.empty_like(q)
        scale_factor = scale if scale is not None else 1 / math.sqrt(q.size(-1))
        q_split_size = 32
        kv_split_size = 32
        # This Loop Level is for parallelization among thread_block
        for paral_index in product(list(range(0, q.size(0))), list(range(0, q.size(1))), list(range(0, q.size(2), q_split_size))):
            q_block_size = q_split_size if ((paral_index[2] + q_split_size) <= q.size(2)) else (q.size(2) - paral_index[2])
            q_block = q[paral_index[0], paral_index[1], paral_index[2]:(paral_index[2]+q_block_size), :].to(torch.float32)
            k_block = k[paral_index[0], paral_index[1], :, :]
            v_block = v[paral_index[0], paral_index[1], :, :]
            # Init temp buffer for kv loop level
            qk_max = torch.empty(q_block_size).fill_(-float('inf'))
            qk_sum = torch.zeros(q_block_size)
            # The result of a block with size [q_block, head_dim]
            acc = torch.zeros_like(q_block)
            # This loop level go through kv_block_szie to keep temp result in share local memory
            kv_iter_size = k.size(2)
            if is_causal:
                kv_iter_size = paral_index[2] + q_block_size  # Skip the block which are fully masked
            for iter_index in range(0, kv_iter_size, kv_split_size):
                kv_block_size = kv_split_size if ((iter_index + kv_split_size) < kv_iter_size) else (kv_iter_size - iter_index)
                _k_block = k_block[iter_index:(iter_index+kv_block_size), :].to(torch.float32)
                _v_block = v_block[iter_index:(iter_index+kv_block_size), :].to(torch.float32)
                # Do first matmul
                attn_weight = torch.matmul(q_block, _k_block.transpose(-2, -1)) # [q_block_size, kv_block_size]
                attn_bias = None
                if (is_causal and (iter_index+kv_block_size) >= kv_iter_size):
                    # For the block which are masked with some elements, apply the mask to the needed elements
                    attn_bias = torch.zeros((q_block_size, kv_block_size), dtype=q.dtype, device=q.device)
                    attn_mask_block = torch.ones(q_block_size, kv_block_size, dtype=torch.bool, device=q.device).tril(diagonal=0)
                    attn_bias.masked_fill_(attn_mask_block.logical_not(), float("-inf"))
                elif attn_mask is not None:
                    attn_bias = torch.zeros((q_block_size, kv_block_size), dtype=q.dtype, device=q.device)
                    attn_mask_block = attn_mask[paral_index[2]:(paral_index[2]+q_block_size), iter_index:(iter_index+kv_block_size)]
                    if attn_mask.dtype == torch.bool:
                        attn_bias.masked_fill_(attn_mask_block.logical_not(), float("-inf"))
                    else:
                        attn_bias = attn_mask_block + attn_bias
                if attn_bias is not None:
                    attn_weight += attn_bias
                
                attn_weight = attn_weight * scale_factor
                # Calculate and update the max value of [q_block_size, kv_block_size]
                block_max = attn_weight.max(dim=-1).values
                qk_max = torch.maximum(qk_max, block_max)
                # Do exp and update the sum value
                exp_scores = torch.exp(attn_weight - qk_max[:, None])
                qk_sum += exp_scores.sum(dim=-1)
                # Do second matmul
                acc += torch.matmul(exp_scores, _v_block) # accumulate [q_block_size, kv_block_size]
            o_ = acc / qk_sum[:, None]
            output[paral_index[0], paral_index[1], paral_index[2]:(paral_index[2]+q_block_size), :] = o_
        return output

lib = torch.library.Library("torch_cuda_extension", "FRAGMENT")
lib.define("extended_add(Tensor a, Tensor b) -> Tensor")  # implement by cuda
lib.define("extended_gemm(Tensor a, Tensor b, str epilogue, bool transpose_B, ScalarType? dtype=None, int api_level=0) -> Tensor")  # implement by cutlass
lib.define("extended_attention(Tensor q, Tensor k, Tensor v, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, *, float? scale=None, int api_level=0) -> Tensor")