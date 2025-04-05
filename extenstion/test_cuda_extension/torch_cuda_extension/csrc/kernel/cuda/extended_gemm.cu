#include <ATen/AccumulateType.h>
#include <torch/library.h>
#include <cuda.h>
#include <cutlass/gemm/device/gemm.h>
#include <c10/cuda/CUDAStream.h>

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

Tensor extended_gemm_kernel(Tensor a, Tensor b, Tensor out) {

    auto a_ptr = a.data_ptr();
    auto b_ptr = b.data_ptr();
    auto out_ptr = out.data_ptr();
    auto N1 = a.size(0);
    auto N2 = a.size(1);

    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    int lda = a.size(1);
    int ldb = b.size(1);
    int ldc = out.size(1);

    //TODO<leslie> assert the scalar_type is float, and the input tensor is 2D

    _extended_gemm_kernel((float*)a_ptr, (float*)b_ptr, (float*)out_ptr, M, N, K, lda, ldb, ldc);

    return out;
}

} // namespace native
} // namespace at
