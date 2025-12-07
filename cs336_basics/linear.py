import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        variance = 2 / (in_features + out_features)
        std = variance**0.5
        nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
