#pragma once

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>
#include <cutlass/gemm/threadblock/default_mma_core.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>

namespace at {
namespace native {

template <typename input_dtype, typename output_dtype, bool use_relu, std::enable_if_t<std::is_same_v<input_dtype, Half>, int> =0> // std::enable_if_t<std::is_same_v<input_dtype, Half>, int> =0
void _extended_gemm_kernel_collective_api(
  input_dtype * a_ptr,
  input_dtype * b_ptr,
  output_dtype * out_ptr,
  int M,
  int N,
  int K,
  int lda,
  int ldb,
  int ldc) {
  TORCH_CHECK(false, "Collective API doesn't support sm80");
}

template <typename input_dtype, typename output_dtype, bool use_relu, std::enable_if_t<!std::is_same_v<input_dtype, Half>, int> =0>
void _extended_gemm_kernel_collective_api(
  input_dtype * a_ptr,
  input_dtype * b_ptr,
  output_dtype * out_ptr,
  int M,
  int N,
  int K,
  int lda,
  int ldb,
  int ldc) {
  TORCH_CHECK(false, "Collective API doesn't support sm80");
}

} // namespace native
} // namespace at
