#pragma once

namespace at {
namespace native {

TORCH_API Tensor extended_gemm_kernel(Tensor a, Tensor b, Tensor out, std::string_view epilogue, bool transpose_B);

} // namespace native
} // namespace at
