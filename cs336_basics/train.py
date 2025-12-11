import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional
import math

def cross_entroy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    max_logit = inputs.max(dim=-1, keepdim=True).values
    inputs -= max_logit
    batch_size = inputs.shape[0]
    correct_logits = inputs[torch.arange(batch_size), targets]

    losses = -correct_logits + torch.log(torch.sum(torch.exp(inputs), dim=-1))
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
                lr_t = lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
                p.data -= lr_t * m / (v.sqrt() + eps)
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v