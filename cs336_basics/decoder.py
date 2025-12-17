from cs336_basics.bpe import Tokenizer
from cs336_basics.transformer import TransformerLM, softmax
from cs336_basics.utils import *

import argparse
import json
import torch
from pathlib import Path
import torch.nn.functional as F


def load(conf: dict) -> tuple[TransformerLM, Tokenizer]:
    device = resolve_device(conf["device"])
    model_dtype_str = conf.get("model_dtype", "float32")
    model_dtype = torch_dtype_from_str(model_dtype_str) or torch.float32
    model = TransformerLM(
        vocab_size=conf["vocab_size"],
        context_length=conf["context_length"],
        d_model=conf["d_model"],
        num_layers=conf["num_layers"],
        num_heads=conf["num_heads"],
        d_ff=conf["d_ff"],
        rope_theta=conf["rope_theta"],
        device=device,
        dtype=model_dtype,
    )
    obj = torch.load(resolve_path(conf["model_path"]))
    model.load_state_dict(obj["model"])

    vocab_path = resolve_path(conf["vocab"])
    merges_path = resolve_path(conf["merges"])
    tokenizer = Tokenizer.from_files(str(vocab_path), str(merges_path), conf["special_tokens"])

    return (model, tokenizer)

def nucleus_sampling(logits: Tensor, threshold: float):
    probs = softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    sum_prob = 0
    cutoff_idx = len(sorted_probs)
    for i, prob in enumerate(sorted_probs):
        sum_prob += prob.item()
        if sum_prob >= threshold:
            cutoff_idx = i + 1
            break
    sorted_probs = sorted_probs[:cutoff_idx]
    sorted_indices = sorted_indices[:cutoff_idx]
    sorted_probs = sorted_probs / sorted_probs.sum()
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices[next_token].item()


def decode(model: torch.nn.Module, tokenizer: Tokenizer, prompt: str, token_limit: int, temperature: float, threshold: float) -> str:
    model.eval()
    tokens: list[int] = tokenizer.encode(prompt)
    # RoPE/attention are defined for a fixed context length; keep a sliding window.
    context_length = getattr(model, "context_length", None)
    if not isinstance(context_length, int) or context_length <= 0:
        raise ValueError("Model is missing a valid `context_length` attribute; cannot decode safely.")

    while len(tokens) < token_limit:
        window = tokens[-context_length:]
        x = torch.tensor(window, dtype=torch.long, device=next(model.parameters()).device)[None, :]
        logits = model(x)[0, -1]
        logits = logits / temperature
        next_token = nucleus_sampling(logits, threshold)
        tokens.append(next_token)
        if next_token == 0:
            break
    return tokenizer.decode(tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default=CONF_DIR / "config.json")
    parser.add_argument("--token_limit", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    conf = get_conf(args.conf)
    model, tokenizer = load(conf)

    prompt = input("Prompt> ")

    output = decode(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        token_limit=200,
        temperature=0.8,
        threshold=0.95,
    )

    print(output)
    