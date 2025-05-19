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
#include "extended_gemm.h"
#include "extended_gemm_collective_api.h"

namespace at {
namespace native {

void _extended_gemm_kernel(float * a_ptr, float * b_ptr, float * out_ptr, int M, int N, int K, int lda, int ldb, int ldc) {
  // Option 1: 创建 CUDA 流
  // cudaStream_t stream;
  // cudaStreamCreate(&stream);

  // Option 2: Use the CUDA Stream from PyTorch
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  using Gemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor
  >;

  Gemm gemm_op;
  float alpha = 1.0;
  float beta = 0.0;
  Gemm::Arguments args(
    {M, N, K},
    {a_ptr, lda},
    {b_ptr, ldb},
    {out_ptr, ldc},
    {out_ptr, ldc},
    {alpha, beta});
  cutlass::Status status = gemm_op(args);

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM 计算失败!" << std::endl;
  }

  // // 同步 CUDA Stream
  // cudaStreamSynchronize(stream);
  // cudaStreamDestroy(stream);
}

__global__ void _extended_gemm_block_naive_kernel(float * a_ptr, float * b_ptr, float * out_ptr, int M, int N, int K, int lda, int ldb, int ldc) {
    // naive cuda implementation, each block calculate a output element
    auto m = blockIdx.x;
    auto n = threadIdx.x;
    if (m < M && n < N) {
      out_ptr[m * ldc + n] = 0.0;
      for (int k=0; k<K; k++) {
        out_ptr[m * ldc + n] += a_ptr[m * lda + k] * b_ptr[k * ldb + n];
      }
    }
}

template <typename T, typename T2, int kTileM, int kTileN, int kTileK, typename TiledMMA, bool use_relu>
__global__ void _extended_gemm_block_cutlass_naive_kernel(
  T * a_ptr,
  T * b_ptr,
  T2 * out_ptr,
  int M,
  int N,
  int K,
  int lda,
  int ldb,
  int ldc) {
    // 构造CUTE Tensor, size 是总的Tensor    
    cute::Tensor A = cute::make_tensor(cute::make_gmem_ptr(a_ptr), cute::make_shape(M, K), cute::make_stride(K, cute::Int<1>{}));
    cute::Tensor B = cute::make_tensor(cute::make_gmem_ptr(b_ptr), cute::make_shape(N, K), cute::make_stride(K, cute::Int<1>{}));  // Column Major
    cute::Tensor C = cute::make_tensor(cute::make_gmem_ptr(out_ptr), cute::make_shape(M, N), cute::make_stride(N, cute::Int<1>{}));

    // 当前block 线程组 要处理的 Tensor Tile 
    int ix = blockIdx.x;  // N 维度，因为 定义 grid(grid_n, grid_m) 
    int iy = blockIdx.y;  // M 维度

    // gA(kTileM, kTileK, num_tile_k)
    // gB(kTileN, kTileK, num_tile_k)
    // gC(kTileM, kTileN) 
    cute::Tensor gA = cute::local_tile(A, cute::make_tile(cute::Int<kTileM>{}, cute::Int<kTileK>{}), cute::make_coord(iy, cute::_));
    cute::Tensor gB = cute::local_tile(B, cute::make_tile(cute::Int<kTileN>{}, cute::Int<kTileK>{}), cute::make_coord(ix, cute::_));
    cute::Tensor gC = cute::local_tile(C, cute::make_tile(cute::Int<kTileM>{}, cute::Int<kTileN>{}), cute::make_coord(iy, ix));

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    // MMA 表示 tiled_mma 单条指令要用到的元素个数
    //   1. refer to: https://github.com/NVIDIA/cutlass/blob/5e497243f7ad13a2aa842143f9b10bbb23d98292/media/docs/cpp/cute/0x_gemm_tutorial.md#tiledmma
    //   2. https://github.com/NVIDIA/cutlass/blob/5e497243f7ad13a2aa842143f9b10bbb23d98292/media/docs/cpp/cute/0t_mma_atom.md#type-aliases
    //      就是 汇编代码 这里的 A，B, C, D 寄存器的大小 对应的元素的个数
    //      以 SM80_16x8x16_F32F16F16F32_TN 为例子：https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm80.hpp
    //      print(tAgA) 看到 MMA 是(2, 2, 2) 表示 这个线程 会处理 8 = 16*16/32 (M*K/num_thread_of_warp)个 A 的元素，
    //        对应了汇编代码中的 a0-a3 x 32bit 寄存器 共 8 * fp16
    //      print(tBgB) 看到 MMA 是(2, 2) 表示 这个线程 会处理 4 = 8*16/32 (N*K/num_thread_of_warp)个 B 的元素，
    //        对应了汇编代码中的 b0-b1 x 32bit 寄存器 共 4 * fp16
    //      print(tCgC) 看到 MMA 是(2, 2) 表示 这个线程 会处理 4 = 8*16/32 (N*K/num_thread_of_warp)个 C 的元素，
    //        对应了汇编代码中的 c0-c3 x 32bit 寄存器 共 4 * fp32
    //      A, B, C 矩阵 每个线程具体操作的元素排布: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float
    //      * tAgA gmem_ptr[16b](0x7f5ff3208000) o ((_2,_2,_2),_1,_2,4):((_1,1024,_8),_0,_16,_32)
    //        * 先看 size
    //          * 首先 (_2,_2,_2) 每个线程 分到 8个元素，4 组 每组 2个 {a0, a1}, {a2, a3}, {a4, a5}, {a6, a7}
    //          * 再看 _1, kTileM (32) 需要 1个 tiled_mma (32)
    //          * 再看 _2, kTileK (32) 需要 2个 tiled_mma (16)
    //          * 再看 4 (注意这个4是动态的，和真实K大小相关), 整个K 是128， 需要4个 kTileK
    //        * 再看 stride
    //          * (_1,1024,_8)
    //             * _1 因为 a0 a1 的间隔是 1
    //             * 1024 （注意是动态的 和K相关），a0 和 a2 差了 8行，每行K(128) 个元素，1024 = 8 * K （128）
    //             * _8, a0 和 a4 之间的间隔是8
    //          * _0, MMA_M 的size 就是1，不需要stride
    //          * _16, MMA_K 两个 tiled_mma 之间 沿着 K 维度 间隔了 16个元素
    //          * _32, 两个 kTileK 之间 沿着 K 维度间隔了 32个元素
    // MMA_M, MMA_K 表示 kTileM, kTileK 按照 tiled_mma 划分需要计算的次数
    // num_tile_k 表示一共有多少个 kTileK
    auto tAgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)
    auto tBgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)
    auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

    // if (cute::thread0()) {
    //   printf("\n");
    //   print(tAgA);
    //   printf("\n");
    //   print(tBgB);
    //   printf("\n");
    //   print(tCgC);
    //   printf("\n");
    // }

    // 返回寄存器声明
    auto tArA = thr_mma.partition_fragment_A(gA(cute::_, cute::_, 0));  // (MMA, MMA_M, MMA_K)
    auto tBrB = thr_mma.partition_fragment_B(gB(cute::_, cute::_, 0));  // (MMA, MMA_N, MMA_K)
    auto tCrC = thr_mma.partition_fragment_C(gC(cute::_, cute::_));     // (MMA, MMA_M, MMA_N)
  
    // set to zero
    cute::clear(tCrC);

    int num_tile_k = cute::size<2>(gA);
    #pragma unroll 1
    for(int itile = 0; itile < num_tile_k; ++itile) {
      cute::copy(tAgA(cute::_, cute::_, cute::_, itile), tArA);
      cute::copy(tBgB(cute::_, cute::_, cute::_, itile), tBrB);

      cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
    }

    if (use_relu) {
      // 手动遍历并应用 ReLU
      // cute::size(tCrC) 是 8 = （kTileM * kTileN） / threadIdx.x
      // 所以应该就表示了 当前 threadIdx.x 要处理的元素
      CUTE_UNROLL
      for (int i = 0; i < cute::size(tCrC); ++i) {
          T2 val = tCrC(i);  // 取出当前值
          tCrC(i) = (val > T2(0)) ? val : T2(0);  // 应用 ReLU 操作
      }

      __syncthreads();
    }

    cute::copy(tCrC, tCgC);
}

template <typename input_dtype, typename output_dtype, bool use_relu, std::enable_if_t<!std::is_same_v<input_dtype, Half>, int> =0>
void _extended_gemm_kernel_low_level_api(
  input_dtype * a_ptr,
  input_dtype * b_ptr,
  output_dtype * out_ptr,
  int M,
  int N,
  int K,
  int lda,
  int ldb,
  int ldc) {
  TORCH_CHECK(false, "None Half input not support yet");
}


template <typename input_dtype, typename output_dtype, bool use_relu, std::enable_if_t<std::is_same_v<input_dtype, Half>, int> =0> // std::enable_if_t<std::is_same_v<input_dtype, Half>, int> =0
void _extended_gemm_kernel_low_level_api(
  input_dtype * a_ptr,
  input_dtype * b_ptr,
  output_dtype * out_ptr,
  int M,
  int N,
  int K,
  int lda,
  int ldb,
  int ldc) {
  constexpr int kTileM = 32;
  constexpr int kTileN = 32;
  constexpr int kTileK = 32;
  int grid_m = M / kTileM;
  int grid_n = N / kTileN;

  using mma_op = cute::SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = cute::MMA_Traits<mma_op>;
  using mma_atom = cute::MMA_Atom<mma_traits>;

  // mma_op will calculate MNK as 16x8x16，tensor core 作用在warp上，所以是一个warp的32线程 的计算size
  // 创建 tiled mma 通过 make_layout 可以进一步扩展  tiled_mma 计算size
  // param: thr layout 表示可以有更多的 warp(线程 来参与 mma计算)， 这里就是 M 维度是2个，N 维度是2个
  // param: val layout (**重要** 最新的code 下面 val layout 的参数被移除了) 表示 在 warp 内部，可以重复参与的mma 计算，这里表示 沿着 N 维度重复2次
  // param: 最后一个参数是 Permutations 被保留了： 
  //   refer to: https://github.com/NVIDIA/cutlass/blob/5e497243f7ad13a2aa842143f9b10bbb23d98292/include/cute/atom/mma_atom.hpp#L207
  //   refer to: https://zhuanlan.zhihu.com/p/28168438940
  // 综上：
  // 一共有 4 个 warp，128个线程参与计算, size(MMA{}) 返回的也是这个结构 (32, cute::_2, cute::_2, cute::_1)
  using MMA = decltype(make_tiled_mma(mma_atom{}, 
                      make_layout(cute::Shape<cute::_2, cute::_2, cute::_1>{}))); // thr layout
                      // make_layout(cute::Shape<cute::_1, cute::_2, cute::_1>{}))); // val layout has been removed


  using mma_op_fp32 = cute::SM80_16x8x16_F32F16F16F32_TN;
  using mma_traits_fp32 = cute::MMA_Traits<mma_op_fp32>;
  using mma_atom_fp32 = cute::MMA_Atom<mma_traits_fp32>;
  using MMA_fp32 = decltype(make_tiled_mma(mma_atom_fp32{}, 
                      make_layout(cute::Shape<cute::_2, cute::_2, cute::_1>{}), // thr layout
                      cute::Tile<cute::_32, cute::_16, cute::_16>{})); // permutation

  dim3 grid(grid_n, grid_m);
  dim3 block;
  using T = cute::half_t;
  using T2 = float;
  if constexpr (std::is_same_v<output_dtype, at::Half>) {
    block = dim3(size(MMA{}));
    _extended_gemm_block_cutlass_naive_kernel<T, T, kTileM, kTileN, kTileK, MMA, use_relu><<<grid, block>>>(
      reinterpret_cast<T*>(a_ptr), reinterpret_cast<T*>(b_ptr), reinterpret_cast<T*>(out_ptr), M, N, K, lda, ldb, ldc
    );
  } else {
    block = dim3(size(MMA_fp32{}));
    _extended_gemm_block_cutlass_naive_kernel<T, T2, kTileM, kTileN, kTileK, MMA_fp32, use_relu><<<grid, block>>>(
      reinterpret_cast<T*>(a_ptr), reinterpret_cast<T*>(b_ptr), (T2*)out_ptr, M, N, K, lda, ldb, ldc
    );
  }

}

Tensor extended_gemm_kernel(
  Tensor a,
  Tensor b,
  Tensor out,
  std::string_view epilogue,
  bool transpose_B,
  int64_t api_level) {
    if (epilogue == "none" && !transpose_B && api_level == 0) {
      // High level API not support epilogue
      // TODO<leslie> assert the scalar_type is float, and the input tensor is 2D
      auto a_ptr = a.data_ptr();
      auto b_ptr = b.data_ptr();
      auto out_ptr = out.data_ptr();

      int M = a.size(0);
      int K = a.size(1);
      int N = b.size(1);

      int lda = a.size(1);
      int ldb = b.size(1);
      int ldc = out.size(1);

      _extended_gemm_kernel((float*)a_ptr, (float*)b_ptr, (float*)out_ptr, M, N, K, lda, ldb, ldc);
    } else if (api_level == 1) {
      // Collective API
      int M = a.size(0);
      int K = a.size(1);
      int N = b.size(0);

      int lda = a.size(1);
      int ldb = b.size(1);
      int ldc = out.size(1);
      // std::cout<<std::is_same_v<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Half>, Half><<std::endl;
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::BFloat16, at::ScalarType::Half, out.scalar_type(),
          "_extended_gemm_kernel_collective_api_kernel_impl",
          [&] { 
            // std::cout<<std::is_same_v<scalar_t, Half><<std::endl;
            Half* a_ptr = a.data_ptr<Half>();
            Half* b_ptr = b.data_ptr<Half>();
            scalar_t* out_ptr = out.data_ptr<scalar_t>();
            if (epilogue == "relu") {
              _extended_gemm_kernel_collective_api<Half, scalar_t, true>(a_ptr, b_ptr, out_ptr, M, N, K, lda, ldb, ldc);
            } else {
              _extended_gemm_kernel_collective_api<Half, scalar_t, false>(a_ptr, b_ptr, out_ptr, M, N, K, lda, ldb, ldc);
            }
          });
    
    }
    else {
      // A is: M x K
      // B is: N x K
      // C is: M x N
      TORCH_CHECK(transpose_B, "for cute api, transpose_B must be true");

      int M = a.size(0);
      int K = a.size(1);
      int N = b.size(0);

      int lda = a.size(1);
      int ldb = b.size(1);
      int ldc = out.size(1);
      // std::cout<<std::is_same_v<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Half>, Half><<std::endl;
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::BFloat16, at::ScalarType::Half, out.scalar_type(),
          "_extended_gemm_kernel_low_level_api_kernel_impl",
          [&] { 
            // std::cout<<std::is_same_v<scalar_t, Half><<std::endl;
            at::Half* a_ptr = a.data_ptr<at::Half>();
            at::Half* b_ptr = b.data_ptr<at::Half>();
            scalar_t* out_ptr = out.data_ptr<scalar_t>();
            if (epilogue == "relu") {
              _extended_gemm_kernel_low_level_api<at::Half, scalar_t, true>(a_ptr, b_ptr, out_ptr, M, N, K, lda, ldb, ldc);
            } else {
              _extended_gemm_kernel_low_level_api<at::Half, scalar_t, false>(a_ptr, b_ptr, out_ptr, M, N, K, lda, ldb, ldc);
            }
          });
    }

    return out;
}

} // namespace native
} // namespace at
