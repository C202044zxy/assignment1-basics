import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        variance = 2 / (in_features + out_features)
        std = variance**0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        sqr = x**2
        rms = torch.sqrt(sqr.mean(-1, keepdim=True) + self.eps)
        x_normed = x / rms * self.weight

        return x_normed.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x1 = x1 * torch.sigmoid(x1)
        x2 = self.w3(x)
        return self.w2(x1 * x2)


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        i = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)[:, None]
        j = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)[None, :]
        theta_ij = i / (theta ** (j / d_k))
        # self.cos = torch.cos(theta_ij)
        # self.sin = torch.sin(theta_ij)
        self.register_buffer("cos", torch.cos(theta_ij), persistent=False)
        self.register_buffer("sin", torch.sin(theta_ij), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        token_positions = token_positions.to(dtype=torch.long)
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

        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

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
        return self.output_proj(multi_head)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = SelfAttention(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        y = x + self.attn(self.ln1(x), torch.arange(x.shape[-2], device=x.device))
        return y + self.ffn(self.ln2(y))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
