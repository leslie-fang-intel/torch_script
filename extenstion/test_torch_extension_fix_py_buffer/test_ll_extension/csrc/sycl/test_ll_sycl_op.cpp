#include <ATen/AccumulateType.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <torch/library.h>
#include "test_ll_sycl_kernel.h"

using namespace at::native::onednn;

namespace at {
namespace native {
namespace xpu {

static Tensor toy_test_sycl_ll(Tensor act, Tensor weight) {
  return testll_kernel(act, weight);
}

TORCH_LIBRARY_IMPL(torchll, XPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("toy_test_sycl_ll"), TORCH_FN(toy_test_sycl_ll));
}

} // namespace xpu
} // namespace native
} // namespace at