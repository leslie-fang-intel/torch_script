#pragma once

namespace at {
namespace native {

TORCH_API Tensor extended_attention_kernel(Tensor q, Tensor k, Tensor v, std::optional<Tensor> attn_mask, double dropout_p, bool is_causal, std::optional<double> scale);

} // namespace native
} // namespace at
