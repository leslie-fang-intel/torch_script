#include <ATen/AccumulateType.h>
#include <torch/library.h>
#include <ATen/functorch/BatchRulesHelper.h>
#include "kernel/cuda/extended_add.h"
#include "kernel/cuda/extended_gemm.h"
#include "kernel/cuda/extended_attention.h"

namespace at {
namespace native {

Tensor extended_add(Tensor a, Tensor b) {
  Tensor out = at::empty_like(a);
  extended_add_kernel(a, b, out);
  return out;
}

Tensor extended_gemm(
  Tensor a,
  Tensor b,
  std::string_view epilogue,
  bool transpose_B,
  std::optional<ScalarType> output_dtype,
  int64_t api_level = 0) {
  // api_level: 0 - GEMM API; 1 - collective API; 2 - Cute API;
  if (!output_dtype.has_value()) {
    output_dtype = a.scalar_type();
  }
  Tensor out = at::empty({a.size(0), b.size(1)}, a.options().dtype(output_dtype));
  if (transpose_B) {
    // B is N * K
    out = at::empty({a.size(0), b.size(0)}, a.options().dtype(output_dtype));
  }
  extended_gemm_kernel(a, b, out, epilogue, transpose_B, api_level);
  return out;
}

Tensor extended_attention(
  Tensor q,
  Tensor k,
  Tensor v,
  std::optional<Tensor> attn_mask = std::nullopt,
  double dropout_p = 0.0,
  bool is_causal = false,
  std::optional<double> scale = std::nullopt,
  int64_t api_level = 0) {
  // api_level: 0 - Reference; 1 - Flash Attention; 2 - Cute API;
  TORCH_CHECK(api_level == 2, "support api_level of 2");
  Tensor out = at::empty_like(q);
  extended_attention_kernel(q, k, v, attn_mask, dropout_p, is_causal, scale, out);
  return out;
}

TORCH_LIBRARY_IMPL(torch_cuda_extension, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("extended_add"), TORCH_FN(extended_add));
  m.impl(TORCH_SELECTIVE_NAME("extended_gemm"), TORCH_FN(extended_gemm));
  m.impl(TORCH_SELECTIVE_NAME("extended_attention"), TORCH_FN(extended_attention));
}

} // namespace native
} // namespace at
