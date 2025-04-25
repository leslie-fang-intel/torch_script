#include <ATen/AccumulateType.h>
#include <torch/library.h>
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/Dispatch.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>
#include <cutlass/gemm/threadblock/default_mma_core.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include "extended_attention.h"

namespace at {
namespace native {

template <typename input_dtype,
          std::enable_if_t<!std::is_same_v<input_dtype, Half>, int> =0> // std::enable_if_t<std::is_same_v<input_dtype, Half>, int> =0
__global__ void _extended_attention_kernel_ref_impl(
  input_dtype* q_ptr,
  input_dtype* k_ptr,
  input_dtype* v_ptr,
  input_dtype* out_ptr,
  int64_t bs,
  int64_t num_head,
  int64_t q_seq_len,
  int64_t kv_seq_len,
  int64_t head_dim,
  int64_t q_split_size,
  int64_t kv_split_size) {

}

template <typename input_dtype,
          std::enable_if_t<std::is_same_v<input_dtype, Half>, int> =0> // std::enable_if_t<std::is_same_v<input_dtype, Half>, int> =0
__global__ void _extended_attention_kernel_ref_impl(
  input_dtype* q_ptr,
  input_dtype* k_ptr,
  input_dtype* v_ptr,
  input_dtype* out_ptr,
  int64_t bs,
  int64_t num_head,
  int64_t q_seq_len,
  int64_t kv_seq_len,
  int64_t head_dim,
  int64_t q_split_size,
  int64_t kv_split_size) {
  
  extern __shared__ float shared_mem[];

  int q_block_id = blockIdx.x;
  int bs_num_head_id = blockIdx.y;
  int thread_id = threadIdx.x;
  int64_t num_q_block = gridDim.x;

  input_dtype* q_ptr_start = q_ptr + bs_num_head_id * q_seq_len * head_dim + q_block_id * q_split_size * head_dim;
  input_dtype* k_ptr_start = k_ptr + bs_num_head_id * kv_seq_len * head_dim;
  input_dtype* v_ptr_start = v_ptr + bs_num_head_id * kv_seq_len * head_dim;
  input_dtype* out_ptr_start = out_ptr + bs_num_head_id * q_seq_len * head_dim + q_block_id * q_split_size * head_dim;

  auto scale_factor = 1 / sqrtf(head_dim);

  // The first matmul
  int64_t q_block_size = q_split_size;
  if ((q_block_id + 1) * q_split_size > q_seq_len) {
    q_block_size = q_seq_len - q_block_id * q_split_size;
  }

  float* attn_weight = shared_mem;
  float* max_buffer = attn_weight + q_block_size * kv_seq_len;
  float* sum_buffer = max_buffer + q_block_size;

  for (int x = 0; x < q_block_size; ++x) {
    for (int y = 0; y < kv_seq_len; ++y) {
      float logit = 0.0f;
      for (int z = 0; z < head_dim; ++z) {
        logit += __half2float(q_ptr_start[x * head_dim + z]) * __half2float(k_ptr_start[y * head_dim + z]);
      }
      attn_weight[x * kv_seq_len + y] = logit;
    }
  }

  // Scale
  for (int x = 0; x < q_block_size; ++x) {
    for (int y = 0; y < kv_seq_len; ++y) {
      attn_weight[x * kv_seq_len + y] = attn_weight[x * kv_seq_len + y] * scale_factor;
    }
  }

  // Softmax
  // 1. find max
  for (int x = 0; x < q_block_size; ++x) {
    float max = -INFINITY;
    float* row = attn_weight + x * kv_seq_len;
    for (int y = 0; y < kv_seq_len; ++y) {
      if (row[y] > max) {
        max = row[y];
      }
    }
    max_buffer[x] = max;
  }

  // 2. exp and sum
  for (int x = 0; x < q_block_size; ++x) {
    float sum = 0.0;
    float* row = attn_weight + x * kv_seq_len;
    for (int y = 0; y < kv_seq_len; ++y) {
        float exp = expf(row[y] - max_buffer[x]);
        sum += exp;
        row[y] = exp;
    }
    sum_buffer[x] = sum;
  }

  // 3. reduce sum
  for (int x = 0; x < q_block_size; ++x) {
    float* row = attn_weight + x * kv_seq_len;
    for (int y = 0; y < kv_seq_len; ++y) {
        row[y] = row[y] / sum_buffer[x];
    }
  }

  // The second matmul
  for (int x = 0; x < q_block_size; ++x) {
    for (int z = 0; z < head_dim; ++z) {
      float logit = 0.0f;
      for (int y = 0; y < kv_seq_len; ++y) {
        logit += attn_weight[x * kv_seq_len + y] * __half2float(v_ptr_start[z + y * head_dim]);
      }
      out_ptr_start[x * head_dim + z] = logit;
    }
  }
}

template <typename input_dtype,
          std::enable_if_t<!std::is_same_v<input_dtype, Half>, int> =0> // std::enable_if_t<std::is_same_v<input_dtype, Half>, int> =0
__global__ void _extended_attention_kernel_impl(
  input_dtype* q_ptr,
  input_dtype* k_ptr,
  input_dtype* v_ptr,
  input_dtype* out_ptr,
  int64_t bs,
  int64_t num_head,
  int64_t q_seq_len,
  int64_t kv_seq_len,
  int64_t head_dim,
  int64_t q_split_size,
  int64_t kv_split_size) {

}

template <typename input_dtype,
          std::enable_if_t<std::is_same_v<input_dtype, Half>, int> =0> // std::enable_if_t<std::is_same_v<input_dtype, Half>, int> =0
__global__ void _extended_attention_kernel_impl(
  input_dtype* q_ptr,
  input_dtype* k_ptr,
  input_dtype* v_ptr,
  input_dtype* out_ptr,
  int64_t bs,
  int64_t num_head,
  int64_t q_seq_len,
  int64_t kv_seq_len,
  int64_t head_dim,
  int64_t q_split_size,
  int64_t kv_split_size) {

}

Tensor extended_attention_kernel(
  Tensor q,
  Tensor k,
  Tensor v,
  std::optional<Tensor> attn_mask,
  double dropout_p,
  bool is_causal,
  std::optional<double> scale,
  Tensor out) {
  // q: [bs, num_head, seq_q, head_dim]
  // k/v: [bs, num_head, seq_kv, head_dim]
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::BFloat16, at::ScalarType::Half, out.scalar_type(),
    "_extended_attention_kernel_impl",
    [&] { 
      scalar_t* q_ptr = q.data_ptr<scalar_t>();
      scalar_t* k_ptr = k.data_ptr<scalar_t>();
      scalar_t* v_ptr = v.data_ptr<scalar_t>();
      scalar_t* out_ptr = out.data_ptr<scalar_t>();

      int64_t bs = q.size(0);
      int64_t num_head = q.size(1);
      int64_t q_seq_len = q.size(2);
      int64_t kv_seq_len = k.size(2);
      int64_t head_dim = q.size(3);

      bool use_reference = true;

      if (use_reference) {
        int64_t q_split_size = 32;
        int64_t kv_split_size = 32;
        int grid_x = (q_seq_len + q_split_size - 1)/ q_split_size;
        int grid_y = bs * num_head;
        // https://github.com/Dao-AILab/flash-attention/blob/a9a3170fc98cbd22a4cc870937b390f3d483f1eb/csrc/flash_attn/src/flash_fwd_launch_template.h#L64
        dim3 grid(grid_x, grid_y);
        // https://github.com/Dao-AILab/flash-attention/blob/a9a3170fc98cbd22a4cc870937b390f3d483f1eb/csrc/flash_attn/src/flash_fwd_launch_template.h#L91 
        // https://github.com/Dao-AILab/flash-attention/blob/a9a3170fc98cbd22a4cc870937b390f3d483f1eb/csrc/flash_attn/src/kernel_traits.h#L63
        // https://github.com/Dao-AILab/flash-attention/blob/a9a3170fc98cbd22a4cc870937b390f3d483f1eb/csrc/flash_attn/src/flash_fwd_launch_template.h#L177
        dim3 block(1);
        size_t shared_mem_size = q_split_size * kv_seq_len * sizeof(float) + q_split_size * sizeof(float) * 2;
        _extended_attention_kernel_ref_impl<scalar_t><<<grid, block, shared_mem_size>>>(
            q_ptr, k_ptr, v_ptr, out_ptr, bs, num_head, q_seq_len, kv_seq_len, head_dim, q_split_size, kv_split_size
        );
      } else {
        // https://github.com/Dao-AILab/flash-attention/blob/a9a3170fc98cbd22a4cc870937b390f3d483f1eb/csrc/flash_attn/src/flash_fwd_launch_template.h#L64
        int64_t q_split_size = 128;
        int64_t kv_split_size = 128;

        int grid_x = (q_seq_len + q_split_size - 1)/ q_split_size;
        int grid_y = bs * num_head;
        dim3 grid(grid_x, grid_y);
        // https://github.com/Dao-AILab/flash-attention/blob/a9a3170fc98cbd22a4cc870937b390f3d483f1eb/csrc/flash_attn/src/flash_fwd_launch_template.h#L91 
        // https://github.com/Dao-AILab/flash-attention/blob/a9a3170fc98cbd22a4cc870937b390f3d483f1eb/csrc/flash_attn/src/kernel_traits.h#L63
        // https://github.com/Dao-AILab/flash-attention/blob/a9a3170fc98cbd22a4cc870937b390f3d483f1eb/csrc/flash_attn/src/flash_fwd_launch_template.h#L177
        // 这个kernel 就是用了 4 个 warp，一共128 threads
        constexpr int kNWarps = 4; // use 4 warp, 128 threads
        constexpr int kNThreads = kNWarps * 32;
        dim3 block(kNThreads);

        // mma_op will calculate MNK as 16x8x16，tensor core 作用在warp上，所以是一个warp的32线程 的计算size
        // param: thr layout MNK 到 64x8x16
        // param: Permutations MNK 到 64x16x16
        using TiledMma = cute::TiledMMA<
            cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>,
            cute::Layout<cute::Shape<cute::Int<kNWarps>,cute::_1,cute::_1>>,  // 4x1x1 or 8x1x1 thread group
            cute::Tile<cute::Int<16 * kNWarps>, cute::_16, cute::_16>>;

        size_t shared_mem_size = q_split_size * kv_seq_len * sizeof(float) + q_split_size * sizeof(float) * 2;
        _extended_attention_kernel_impl<scalar_t><<<grid, block, shared_mem_size>>>(
            q_ptr, k_ptr, v_ptr, out_ptr, bs, num_head, q_seq_len, kv_seq_len, head_dim, q_split_size, kv_split_size
        );
      }
    });

  return out;
}

} // namespace native
} // namespace at
