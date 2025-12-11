import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

def cross_entroy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    max_logit = inputs.max(dim=-1, keepdim=True).values
    inputs -= max_logit
    batch_size = inputs.shape[0]
    correct_logits = inputs[torch.arange(batch_size), targets]

    losses = -correct_logits + torch.log(torch.sum(torch.exp(inputs), dim=-1))
    return losses.mean()
