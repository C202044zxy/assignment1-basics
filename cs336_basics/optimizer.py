import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional
import math


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    inputs = inputs.reshape(-1, inputs.shape[-1])
    targets = targets.reshape(-1)
    max_logit = inputs.max(dim=-1, keepdim=True).values
    # Avoid materializing a huge exp(inputs) tensor; use logsumexp for stability and memory.
    inputs = inputs - max_logit
    batch_size = inputs.shape[0]
    correct_logits = inputs[torch.arange(batch_size), targets]

    losses = -correct_logits + torch.logsumexp(inputs, dim=-1)
    return losses.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float, betas: tuple[float, float], eps: float = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", 0)
                v = state.get("v", 0)

                grad = p.grad.data
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad * grad
                lr_t = lr * math.sqrt(1 - b2**t) / (1 - b1**t)
                p.data -= lr_t * m / (v.sqrt() + eps)
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v


def lr_cosine_schedule(t: int, a_max: float, a_min: float, t_w: int, t_c: int) -> float:
    if t < t_w:
        return (t / t_w) * a_max
    elif t <= t_c:
        return a_min + 0.5 * (1 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (a_max - a_min)
    else:
        return a_min


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    param_list = [param for param in parameters if param.grad is not None]
    l2_norm_sq = sum((p.grad**2).sum().item() for p in param_list)
    l2_norm = math.sqrt(l2_norm_sq)
    if l2_norm <= max_l2_norm:
        return
    scale = max_l2_norm / (l2_norm + 1e-6)
    for param in param_list:
        param.grad.mul_(scale)
