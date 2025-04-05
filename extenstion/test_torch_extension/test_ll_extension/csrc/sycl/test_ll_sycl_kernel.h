#pragma once

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API Tensor testll_kernel(Tensor act, Tensor weight);

}
} // namespace native
} // namespace at
