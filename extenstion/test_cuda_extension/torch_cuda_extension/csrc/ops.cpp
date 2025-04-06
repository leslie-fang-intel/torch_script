#include <ATen/AccumulateType.h>
#include <torch/library.h>
#include <ATen/functorch/BatchRulesHelper.h>
#include "kernel/cuda/extended_add.h"
#include "kernel/cuda/extended_gemm.h"

namespace at {
namespace native {

Tensor extended_add(Tensor a, Tensor b) {
  Tensor out = at::empty_like(a);
  extended_add_kernel(a, b, out);
  return out;
}

Tensor extended_gemm(Tensor a, Tensor b, std::string_view epilogue, bool transpose_B) {
  Tensor out = at::empty({a.size(0), b.size(1)}, a.options());
  if (transpose_B) {
    // B is N * K
    out = at::empty({a.size(0), b.size(0)}, a.options());
  }
  extended_gemm_kernel(a, b, out, epilogue, transpose_B);
  return out;
}

TORCH_LIBRARY_IMPL(torch_cuda_extension, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("extended_add"), TORCH_FN(extended_add));
  m.impl(TORCH_SELECTIVE_NAME("extended_gemm"), TORCH_FN(extended_gemm));
}

} // namespace native
} // namespace at
