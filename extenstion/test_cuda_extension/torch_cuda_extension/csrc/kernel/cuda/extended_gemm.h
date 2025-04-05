#pragma once

namespace at {
namespace native {

TORCH_API Tensor extended_gemm_kernel(Tensor a, Tensor b, Tensor out);

} // namespace native
} // namespace at
