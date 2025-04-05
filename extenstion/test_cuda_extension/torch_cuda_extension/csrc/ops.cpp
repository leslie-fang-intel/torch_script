#include <ATen/AccumulateType.h>
#include <torch/library.h>
#include <ATen/functorch/BatchRulesHelper.h>
#include "kernel/cuda/extended_add.h"

namespace at {
namespace native {

Tensor extended_add(Tensor a, Tensor b) {
  // return at::add(a, b);
  Tensor out = at::empty_like(a);
  extended_add_kernel(a, b, out);
  return out;
}

TORCH_LIBRARY_IMPL(torch_cuda_extension, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("extended_add"), TORCH_FN(extended_add));
}

} // namespace native
} // namespace at
