#include <ATen/AccumulateType.h>
#include <torch/library.h>
#include <ATen/functorch/BatchRulesHelper.h>
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

template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    using namespace cute;
    static_assert(decltype(cute::size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return cute::max(x, y); }
};

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(cute::Tensor<Engine0, Layout0> const &tensor, cute::Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(cute::size<0>(summary) == cute::size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < cute::size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < cute::size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

template<>
struct Allreduce<2> {
template<typename T, typename Operator> 
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(cute::Tensor<Engine0, Layout0> &dst, cute::Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(cute::size(dst) == cute::size(src));
    #pragma unroll
    for (int i = 0; i < cute::size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(cute::Tensor<Engine0, Layout0> const& tensor, cute::Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(cute::Tensor<Engine0, Layout0> const& tensor, cute::Tensor<Engine1, Layout1> &max){
    using namespace cute;
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(cute::Tensor<Engine0, Layout0> &tensor, cute::Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(cute::size<0>(max) == cute::size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < cute::size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < cute::size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            // The following macro will disable the use of fma.
            // See: https://github.com/pytorch/pytorch/issues/121558 for more details
            // This macro is set in PyTorch and not FlashAttention
            #ifdef UNFUSE_FMA
                tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
            #else
                tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            #endif
        }
    }
}

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(cute::Tensor<Engine0, Layout0> const& tensor, cute::Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}

template <int kNRows>
struct Softmax {
    using TensorT = decltype(cute::make_tensor<float>(cute::Shape<cute::Int<kNRows>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ Softmax() {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        using namespace cute;
        cute::Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(cute::size<0>(scores))::value == kNRows);
        if (Is_first) {
            reduce_max</*zero_init=*/true>(scores, row_max);
            scale_apply_exp2(scores, row_max, softmax_scale_log2);
            reduce_sum</*zero_init=*/true>(scores, row_sum);
        } else {
            // Tensor scores_max_prev = make_fragment_like(row_max);
            // cute::copy(row_max, scores_max_prev);
            // FLASH_NAMESPACE::template reduce_max</*zero_init=*/false>(scores, row_max);
            // // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            // Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
            // static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            // #pragma unroll
            // for (int mi = 0; mi < size(row_max); ++mi) {
            //     float scores_max_cur = !Check_inf
            //         ? row_max(mi)
            //         : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
            //     float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
            //     row_sum(mi) *= scores_scale;
            //     #pragma unroll
            //     for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            // }
            // FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // // We don't do the reduce across threads here since we don't need to use the row_sum.
            // // We do that reduce at the end when we need to normalize the softmax.
            // FLASH_NAMESPACE::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
    };

    template<bool Is_dropout=false, bool Split=false, typename Tensor0>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
        using namespace cute;
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT lse = make_fragment_like(row_sum);
        cute::Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(cute::size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < cute::size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #pragma unroll
            for (int ni = 0; ni < cute::size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
        }
        return lse;
    };
};

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto _convert_type(cute::Tensor<Engine, Layout> const &tensor) {
    using namespace cute;
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <int N>
CUTE_HOST_DEVICE
void _cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void _gemm_gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    using namespace cute;
    CUTE_STATIC_ASSERT_V(cute::size<1>(tCrA) == cute::size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(cute::size<1>(tCrB) == cute::size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(cute::size<2>(tCrA) == cute::size<2>(tCrB));                     // MMA_K
    cute::Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(cute::size<1>(tCsA) == cute::size<1>(tCrA_copy_view));            // M
    cute::Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(cute::size<1>(tCsB) == cute::size<1>(tCrB_copy_view));            // N
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < cute::size<2>(tCrA); ++i) {
        if (i < cute::size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void _copy_copy(TiledCopy tiled_copy, cute::Tensor<Engine0, Layout0> const &S,
                            cute::Tensor<Engine1, Layout1> &D, cute::Tensor<Engine2, Layout2> const &identity_MN,
                            cute::Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    using namespace cute;
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(cute::size<0>(S) == cute::size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(cute::size<1>(S) == cute::size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(cute::size<2>(S) == cute::size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < cute::size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < cute::size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto _convert_layout_acc_Aregs(Layout acc_layout, MMA_traits tiled_mma) {
    using namespace cute;
    using X = Underscore;
    static_assert(decltype(cute::size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void _gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    using namespace cute;
    CUTE_STATIC_ASSERT_V(cute::size<1>(tCrA) == cute::size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(cute::size<1>(tCrB) == cute::size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(cute::size<2>(tCrA) == cute::size<2>(tCrB));                     // MMA_K
    cute::Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(cute::size<1>(tCsB) == cute::size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < cute::size<2>(tCrA); ++i) {
        if (i < cute::size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template <typename input_dtype, typename TiledMMA, typename GmemTiledCopyQKV, int q_split_size, int kv_split_size, int head_dim, typename SmemLayoutQ, typename SmemLayoutKV, typename SmemLayoutVtransposed, typename SmemLayoutVtransposedNoSwizzle, typename GmemTiledCopyO, typename SmemLayoutO,
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
  // int64_t head_dim,
  const float softmax_scale) {
}

template <typename input_dtype, typename TiledMMA, typename GmemTiledCopyQKV, int q_split_size, int kv_split_size, int head_dim, typename SmemLayoutQ, typename SmemLayoutKV, typename SmemLayoutVtransposed, typename SmemLayoutVtransposedNoSwizzle, typename GmemTiledCopyO, typename SmemLayoutO,
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
  // int64_t head_dim,
  const float softmax_scale) {
  using namespace cute;
  extern __shared__ char smem_[];

  int q_block_id = blockIdx.x;
  int bs_id = blockIdx.y;
  int head_id = blockIdx.z;
  int thread_id = threadIdx.x;
  int64_t num_q_block = gridDim.x;

  int n_block_max = cute::ceil_div(kv_seq_len, kv_split_size);
  int n_block = n_block_max - 1;
  int n_block_min = 0;

  int64_t q_row_stride = head_dim;
  // Although we transpose q to (Batch x Q_seq_len  x Num_heads x Dim_per_head)
  // We didn't do contiguous along the num_head dim
  int64_t q_head_stride = q_seq_len * head_dim;
  int64_t k_row_stride = head_dim;
  int64_t k_head_stride = kv_seq_len * head_dim;

  if (cute::thread0()) {
    printf(" ------ n_block is: %d \n", n_block);
    printf(" ------ q_row_stride is: %ld \n", q_row_stride);
    printf(" ------ q_head_stride is: %ld \n", q_head_stride);
    printf(" ------ k_row_stride is: %ld \n", k_row_stride);
    printf(" ------ k_head_stride is: %ld \n", k_head_stride);

    static constexpr int kBlockKSmem = 64;
    static constexpr int kBlockKGmem = 64;
    static constexpr int kSwizzle = 3;
    static constexpr int kBlockM = 128;
    static constexpr int kHeadDim = 64;
  }

  // q
  // slice 1 along bs
  cute::Tensor mQ = make_tensor(
    make_gmem_ptr(reinterpret_cast<cutlass::half_t*>(q_ptr) + bs_id * num_head * q_seq_len * head_dim),
    make_shape(q_seq_len, num_head, head_dim),
    make_stride(q_row_stride, q_head_stride, _1{})
  ); // [1, Q_seq_len, Num_heads, Head_Dim]
  // get slice 1 of num_head, then get tile along Q_seq
  cute::Tensor gQ = local_tile(mQ(_, head_id, _), Shape<Int<q_split_size>, Int<head_dim>>{},
                          make_coord(q_block_id, 0)); // [1, q_slice, 1, head_dim]
  // k
  cute::Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<cutlass::half_t*>(k_ptr) + bs_id * num_head * kv_seq_len * head_dim),
                          make_shape(kv_seq_len, num_head, head_dim),
                          make_stride(k_row_stride, k_head_stride, _1{}));  // [1, kv_seq_len, Num_heads, Head_Dim]
  cute::Tensor gK = local_tile(mK(_, head_id, _), Shape<Int<kv_split_size>, Int<head_dim>>{},
                          make_coord(_, 0));  // (kv_slice, head_dim, num_of_kv_blocks)
  // v
  cute::Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<cutlass::half_t*>(v_ptr) + bs_id * num_head * kv_seq_len * head_dim),
                          make_shape(kv_seq_len, num_head, head_dim),
                          make_stride(k_row_stride, k_head_stride, _1{}));
  cute::Tensor gV = local_tile(mV(_, head_id, _), Shape<Int<kv_split_size>, Int<head_dim>>{},
                          make_coord(_, 0));  // (kv_slice, head_dim, num_of_kv_blocks)

  cute::Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<cutlass::half_t *>(smem_)), SmemLayoutQ{});
  cute::Tensor sK = make_tensor(sQ.data() + size(sQ), SmemLayoutKV{});
  cute::Tensor sV = make_tensor(sK.data() + size(sK), SmemLayoutKV{});
  cute::Tensor sVt = make_tensor(sV.data(), SmemLayoutVtransposed{});
  cute::Tensor sVtNoSwizzle = make_tensor(sV.data().get(), SmemLayoutVtransposedNoSwizzle{});

  const int tidx = threadIdx.x;

  GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

  cute::Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  cute::Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  cute::Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
  cute::Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  cute::Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
  cute::Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  cute::Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
  cute::Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
  cute::Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

  // Tensor tSgS  = thr_mma.partition_C(gP);

  cute::Tensor cQ = make_identity_tensor(make_shape(cute::size<0>(sQ), cute::size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
  cute::Tensor cKV = make_identity_tensor(make_shape(cute::size<0>(sK), cute::size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
  cute::Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  cute::Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
  
  cute::Tensor tQpQ = make_tensor<bool>(make_shape(cute::size<2>(tQsQ)));
  cute::Tensor tKVpKV = make_tensor<bool>(make_shape(cute::size<2>(tKsK)));

  cute::Tensor acc_o = partition_fragment_C(tiled_mma, cute::Shape<Int<q_split_size>, Int<head_dim>>{});  // MMA, MMA_M, MMA_K

  Softmax<2 * cute::size<1>(acc_o)> softmax;

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, cutlass::half_t>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, cutlass::half_t>;
  auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
  cute::Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  cute::Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  cute::Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  constexpr int n_masking_steps = 1;
  #pragma unroll
  for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
      cute::Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<q_split_size>, Int<kv_split_size>>{});  // (MMA=4, MMA_M, MMA_N)
      clear(acc_s);
      _cp_async_wait<0>();
      __syncthreads();

      _copy_copy<true, true, /*Clear_OOB_MN=*/true>(
          gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV, kv_seq_len - n_block * kv_split_size
      );

      cute::cp_async_fence();

      _gemm_gemm<false>(
          acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
          smem_thr_copy_Q, smem_thr_copy_K
      );

      _cp_async_wait<0>();
      __syncthreads();
      # define M_LOG2E	1.4426950408889634074	/* log_2 e */
      auto softmax_scale_log2 = softmax_scale * M_LOG2E;
      softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/false>(acc_s, acc_o, softmax_scale_log2);

      // // Convert acc_s from fp32 to fp16/bf16
      cute::Tensor rP = _convert_type<cutlass::half_t>(acc_s);

      // // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
      // // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
      cute::Tensor tOrP = make_tensor(rP.data(), _convert_layout_acc_Aregs(rP.layout(), tiled_mma));
      _gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
  }

  // if (cute::thread0()) {
  //   printf("softmax_scale is: %f", softmax_scale);
  // }

  int64_t o_row_stride = head_dim;
  int64_t o_head_stride = q_seq_len * head_dim;
  cute::Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<cutlass::half_t*>(out_ptr) + bs_id * num_head * q_seq_len * head_dim),
                          make_shape(q_seq_len, num_head, head_dim),
                          make_stride(o_row_stride, o_head_stride, _1{}));
  cute::Tensor gO = local_tile(mO(_, head_id, _), cute::Shape<Int<q_split_size>, Int<head_dim>>{},
                          make_coord(q_block_id, 0));  // (kBlockM, kHeadDim)

  cute::Tensor lse = softmax.template normalize_softmax_lse<false>(acc_o, softmax_scale, 1.0);
  cute::Tensor rO = _convert_type<cutlass::half_t>(acc_o);
  cute::Tensor sO = make_tensor(sQ.data(), SmemLayoutO{});    // (SMEM_M,SMEM_N)

  // Copy from acc_o to sQ
  using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, cutlass::half_t>;
  auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  cute::Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
  cute::Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  // Copy from sQ to temp buf tOrO
  GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  cute::Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
  cute::Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

   __syncthreads();

  cute::Tensor tOrO = make_tensor<cutlass::half_t>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);


  // Construct identity layout for sO
  cute::Tensor cO = make_identity_tensor(make_shape(cute::size<0>(sO), cute::size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
  // Repeat the partitioning with identity layouts
  cute::Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  cute::Tensor tOpO = make_tensor<bool>(make_shape(cute::size<2>(tOgO)));

  __syncthreads();

  // Copy from temp buf tOrO to global buf
  _copy_copy<true, true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
      gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, q_seq_len - q_block_id * q_split_size
  );
}

Tensor extended_attention_kernel(
  Tensor q,
  Tensor k,
  Tensor v,
  std::optional<Tensor> attn_mask,
  double dropout_p,
  bool is_causal,
  std::optional<double> scale) {
  bool use_reference = false;
  Tensor out = at::empty_like(q);

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::BFloat16, at::ScalarType::Half, out.scalar_type(),
    "__extended_attention_kernel__impl",
    [&] { 
      if (use_reference) {
        // q: [bs, num_head, seq_q, head_dim]
        // k/v: [bs, num_head, seq_kv, head_dim]
        scalar_t* q_ptr = q.data_ptr<scalar_t>();
        scalar_t* k_ptr = k.data_ptr<scalar_t>();
        scalar_t* v_ptr = v.data_ptr<scalar_t>();
        scalar_t* out_ptr = out.data_ptr<scalar_t>();

        int64_t bs = q.size(0);
        int64_t num_head = q.size(1);
        int64_t q_seq_len = q.size(2);
        int64_t kv_seq_len = k.size(2);
        int64_t head_dim = q.size(3);

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

        printf("---- hit here ----");

        // q: [bs, num_head, seq_q, head_dim]
        // k/v: [bs, num_head, seq_kv, head_dim]
        int64_t bs = q.size(0);
        int64_t num_head = q.size(1);
        int64_t q_seq_len = q.size(2);
        int64_t kv_seq_len = k.size(2);
        int64_t head_dim = q.size(3);

        TORCH_CHECK(head_dim == 64);
        constexpr int _head_dim = 64;

        // Query -> Query(Batch x Q_seq_len  x Num_heads x Dim_per_head)
        // Key   -> Key  (Batch x KV_seq_len x Num_heads x Dim_per_head)
        // Value -> Value(Batch x KV_seq_len x Num_heads x Dim_per_head)
        Tensor q_t = q.transpose(1, 2);
        Tensor k_t = k.transpose(1, 2);
        Tensor v_t = v.transpose(1, 2);

        Tensor out_t = out.transpose(1, 2);

        scalar_t* q_ptr = q_t.data_ptr<scalar_t>();
        scalar_t* k_ptr = k_t.data_ptr<scalar_t>();
        scalar_t* v_ptr = v_t.data_ptr<scalar_t>();
        scalar_t* out_ptr = out_t.data_ptr<scalar_t>();

        constexpr int q_split_size = 128;
        constexpr int kv_split_size = 128;

        int grid_x = (q_seq_len + q_split_size - 1)/ q_split_size;
        int grid_y = bs;
        int grid_z = num_head;
        dim3 grid(grid_x, grid_y, grid_z);
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

        const auto softmax_scale_symbol = scale.has_value()
            ? scale.value()
            : (c10::SymFloat(1.0) / (c10::SymFloat(q.sym_size(-1)).sqrt()));
        const float softmax_scale = c10::SymFloat(softmax_scale_symbol).expect_float();

        // size_t shared_mem_size = q_split_size * kv_seq_len * sizeof(float) + q_split_size * sizeof(float) * 2;
        using namespace cute;
        static constexpr int kBlockKSmem = _head_dim % 64 == 0 ? 64 : 32;
        static constexpr int kBlockKGmem = _head_dim % 128 == 0 ? 128 : (_head_dim % 64 == 0 ? 64 : 32);
        static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

        using SmemLayoutAtomQ = decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                  // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                  cute::Layout<cute::Shape<_8, Int<kBlockKSmem>>,
                          cute::Stride<Int<kBlockKSmem>, _1>>{}));
        using SmemLayoutQ = decltype(tile_to_shape(
            SmemLayoutAtomQ{},
            cute::Shape<Int<q_split_size>, Int<_head_dim>>{}));
        using SmemLayoutKV = decltype(tile_to_shape(
            SmemLayoutAtomQ{},
            Shape<Int<kv_split_size>, Int<_head_dim>>{}));

        // Define the TiledCopy
        static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(cutlass::half_t);
        static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
        using GmemLayoutAtom = cute::Layout<cute::Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                      cute::Stride<Int<kGmemThreadsPerRow>, _1>>;
        using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
        using GmemTiledCopyQKV = decltype(
            make_tiled_copy(Copy_Atom<Gmem_copy_struct, cutlass::half_t>{},
                            GmemLayoutAtom{},
                            cute::Layout<cute::Shape<_1, _8>>{}));  // Val layout, 8 vals per read

        static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(cutlass::half_t);
        static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(cutlass::half_t);
        size_t shared_mem_size = kSmemQSize + kSmemKVSize;
        using SmemLayoutVtransposed = decltype(
          composition(SmemLayoutKV{}, make_layout(Shape<Int<_head_dim>, Int<kv_split_size>>{}, GenRowMajor{})));
        using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

        using GmemTiledCopyO = decltype(
            make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, cutlass::half_t>{},
                            GmemLayoutAtom{},
                            cute::Layout<cute::Shape<_1, _8>>{}));  // Val layout, 8 vals per store

        using SmemLayoutAtomO = decltype(
            composition(Swizzle<kSwizzle, 3, 3>{},
                        cute::Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                              cute::Stride<Int<kBlockKSmem>, _1>>{}));

        using SmemLayoutO = decltype(tile_to_shape(
            SmemLayoutAtomO{},
            cute::Shape<Int<q_split_size>, Int<_head_dim>>{}));

        _extended_attention_kernel_impl<scalar_t, TiledMma, GmemTiledCopyQKV, q_split_size, kv_split_size, _head_dim, SmemLayoutQ, SmemLayoutKV, SmemLayoutVtransposed, SmemLayoutVtransposedNoSwizzle, GmemTiledCopyO, SmemLayoutO><<<grid, block, shared_mem_size>>>(
            q_ptr, k_ptr, v_ptr, out_ptr, bs, num_head, q_seq_len, kv_seq_len, softmax_scale
        );

        // Transpose back
        out = out_t.transpose(1, 2);
      }
    });

  return out;
}

} // namespace native
} // namespace at
