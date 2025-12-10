import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        sqr = x**2
        rms = torch.sqrt(sqr.mean(-1, keepdim=True) + self.eps)
        x_normed = x / rms * self.g

        return x_normed.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.ones(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.ones(d_ff, d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        x1 = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        x1 = x1 * torch.sigmoid(x1)
        x2 = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        return einsum(x1 * x2, self.w2, "... d_ff, d_model d_ff -> ... d_model")


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        i = torch.arange(0, max_seq_len).float()[:, None]
        j = torch.arange(0, d_k, 2).float()[None, :]
        theta_ij = i / (theta ** (j / d_k))
        # self.cos = torch.cos(theta_ij)
        # self.sin = torch.sin(theta_ij)
        self.register_buffer("cos", torch.cos(theta_ij), persistent=False)
        self.register_buffer("sin", torch.sin(theta_ij), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        cos_pos = self.cos[token_positions, :]
        sin_pos = self.sin[token_positions, :]
        q0 = x[..., 0::2]
        q1 = x[..., 1::2]
        return torch.stack((q0 * cos_pos - q1 * sin_pos, q0 * sin_pos + q1 * cos_pos), dim=-1).flatten(-2)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    max_x = x.max(dim, keepdim=True).values
    x = torch.exp(x - max_x)
    return x / x.sum(dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    sqrt_d_k = Q.shape[-1] ** 0.5
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / sqrt_d_k
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    scores = softmax(scores)
    return einsum(scores, V, "... queries keys, ... keys d_v -> ... queries d_v")


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 0,
        theta: float = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        if max_seq_len > 0:
            # use rope
            self.rope = RoPE(theta, self.d_k, max_seq_len, device)

        self.wq = nn.Parameter(torch.ones(d_model, d_model, device=device, dtype=dtype))
        self.wk = nn.Parameter(torch.ones(d_model, d_model, device=device, dtype=dtype))
        self.wv = nn.Parameter(torch.ones(d_model, d_model, device=device, dtype=dtype))
        self.wo = nn.Parameter(torch.ones(d_model, d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        q = einsum(x, self.wq, "... i, j i -> ... j")
        k = einsum(x, self.wk, "... i, j i -> ... j")
        v = einsum(x, self.wv, "... i, j i -> ... j")

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()

        head_outputs = []
        for i in range(0, self.d_model, self.d_k):
            q_head = q[..., i : i + self.d_k]
            k_head = k[..., i : i + self.d_k]
            v_head = v[..., i : i + self.d_k]
            if token_positions is not None:
                q_head = self.rope(q_head, token_positions)
                k_head = self.rope(k_head, token_positions)
            head_outputs.append(scaled_dot_product_attention(q_head, k_head, v_head, mask))

        multi_head = torch.cat(head_outputs, dim=-1)
        return einsum(multi_head, self.wo, "... i, j i -> ... j")
