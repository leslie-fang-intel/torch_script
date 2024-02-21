import torch

def dot_prod_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training
) -> torch.Tensor:
    """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    bs = q.size(0)
    k_len = k.size(-2)
    attn_mask = torch.ones(
        bs, k_len, dtype=torch.bool, device=query.device
    ).tril(diagonal=0)
    scores = torch.matmul(q, k.transpose(-2, -1)) / 3.0
    attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
    scores = scores.masked_fill(attn_mask, -float("inf"))
    weights = torch.nn.functional.softmax(scores, dim=-1)
    weights = torch.nn.functional.dropout(
        weights,
        p=0.4,
        training=training,
        inplace=False,
    )
    return torch.matmul(weights, v)

if __name__ == "__main__":
    dtype = torch.float32
    device = "cpu"
    tensor_shape = (4, 2, 16, 32)
    args1 = [
        torch.randn(tensor_shape, device=device, dtype=dtype),
        torch.randn(tensor_shape, device=device, dtype=dtype),
        torch.randn(tensor_shape, device=device, dtype=dtype),
    ]
    dropout_arg = [False]

    with torch.no_grad():
        res_ref = dot_prod_attention(*(args1 + dropout_arg))

        cfn = torch.compile(dot_prod_attention)
        inductor_res = cfn(*(args1 + dropout_arg))
        print(torch.allclose(res_ref, inductor_res, atol=0.01, rtol=0.01), flush=True)