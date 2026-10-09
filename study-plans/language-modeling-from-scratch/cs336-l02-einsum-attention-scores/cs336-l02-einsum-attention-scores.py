import torch

def attention_scores(q: torch.Tensor, k: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Returns scores of shape (batch, heads, query_length, key_length).
    """

    B, s_q, D = q.shape
    _, s_k, _ = k.shape
    dh = D//num_heads

    q = torch.reshape(q, (B, s_q, num_heads, dh))
    k = torch.reshape(k, (B, s_k, num_heads, dh))

    A = torch.einsum('bihr,bjhr->bhij', q, k) / (dh**0.5)

    return A
    pass
