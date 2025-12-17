from __future__ import annotations

import torch
import json
from pathlib import Path
import argparse
import wandb
import numpy as np
import os

from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import *
from cs336_basics.utils import *
from cs336_basics.bpe import Tokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
conf_DIR = ROOT_DIR / "config"


def get_conf() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default=conf_DIR / "config.json")
    args = parser.parse_args()
    with open(args.conf, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _resolve_path(p: str | os.PathLike) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def _torch_dtype_from_str(dtype: str | None) -> torch.dtype | None:
    if dtype is None:
        return None
    if isinstance(dtype, str):
        # Accept common strings like "float32", "bfloat16", etc.
        t = getattr(torch, dtype, None)
        return t if isinstance(t, torch.dtype) else None
    return None


def load_memmap(path: Path, dtype: str) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(f"Memmap file not found {path}")
    return np.memmap(path, dtype=dtype, mode="r")


def _tokenize_text_to_bin(
    *,
    input_txt: Path,
    output_bin: Path,
    tokenizer: Tokenizer,
    dtype: str,
    flush_tokens: int = 200_000,
) -> Path:
    """
    Stream-tokenize a text corpus into a flat binary array of token ids.
    The output can be memory-mapped with np.memmap(output_bin, dtype=dtype, mode="r").
    """
    np_dtype = np.dtype(dtype)
    # If a previous run was interrupted, we may have a partially-written file. Treat
    # any size mismatch as corrupt and rebuild.
    if output_bin.exists():
        size = output_bin.stat().st_size
        if size > 0 and size % np_dtype.itemsize == 0:
            return output_bin
        output_bin.unlink(missing_ok=True)
    output_bin.parent.mkdir(parents=True, exist_ok=True)
    if not input_txt.exists():
        raise FileNotFoundError(f"Text file not found {input_txt}")

    tmp = output_bin.with_suffix(output_bin.suffix + ".tmp")
    buf: list[int] = []
    with input_txt.open("r", encoding="utf-8", errors="ignore") as f_in, tmp.open("wb") as f_out:
        for tok in tokenizer.encode_iterable(f_in):
            buf.append(tok)
            if len(buf) >= flush_tokens:
                np.asarray(buf, dtype=np_dtype).tofile(f_out)
                buf.clear()
        if buf:
            np.asarray(buf, dtype=np_dtype).tofile(f_out)

    os.replace(tmp, output_bin)
    return output_bin


def _load_token_dataset(path: Path, tokenizer: Tokenizer, dtype: str) -> np.memmap:
    """
    Load a tokenized dataset as a memory-mapped 1D array of token ids.

    - If `path` is a `.txt`, it will be streamed through the tokenizer once and cached as `*.bin`.
    - If `path` is already a binary file (e.g. `.bin`), it will be memory-mapped directly.
    """
    if path.suffix.lower() == ".txt":
        np_dtype = np.dtype(dtype)
        token_path = path.with_suffix(f".{np_dtype.name}.bin")
        token_path = _tokenize_text_to_bin(input_txt=path, output_bin=token_path, tokenizer=tokenizer, dtype=dtype)
        return load_memmap(token_path, dtype)
    return load_memmap(path, dtype)


def estimate_loss(
    model: torch.nn.Module,
    train_data: npt.NDArray,
    val_data: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_runs: int,
):
    model.eval()
    out: dict[str, float] = {}
    for name, dataset in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(eval_runs):
            xb, yb = get_batch(dataset, batch_size, context_length, device)
            logits = model(xb)
            loss = cross_entropy(logits, yb)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def main():
    # 1. read conf and load data. Do tokenization.
    conf = get_conf()
    batch_size = conf["batch_size"]
    context_length = conf["context_length"]
    device = conf["device"]
    data_dtype = conf.get("data_dtype", conf.get("dtype", "uint16"))
    model_dtype_str = conf.get("model_dtype", "float32")
    model_dtype = _torch_dtype_from_str(model_dtype_str) or torch.float32

    train_data_path = _resolve_path(conf["train_data"])
    val_data_path = _resolve_path(conf["val_data"])
    vocab_path = _resolve_path(conf["vocab"])
    merges_path = _resolve_path(conf["merges"])
    tokenizer = Tokenizer.from_files(str(vocab_path), str(merges_path), conf["special_tokens"])
    train_data = _load_token_dataset(train_data_path, tokenizer, data_dtype)
    val_data = _load_token_dataset(val_data_path, tokenizer, data_dtype)
    print(f"load data succ")

    # 2. init the model and optimizer
    model = TransformerLM(
        vocab_size=conf["vocab_size"],
        context_length=context_length,
        d_model=conf["d_model"],
        num_layers=conf["num_layers"],
        num_heads=conf["num_heads"],
        d_ff=conf["d_ff"],
        rope_theta=conf["rope_theta"],
        device=device,
        dtype=model_dtype,
    )
    optimizer = AdamW(
        params=model.parameters(),
        lr=conf["lr"],
        weight_decay=conf["weight_decay"],
        betas=(conf["beta1"], conf["beta2"]),
        eps=conf["eps"],
    )

    # 3. start training loop
    # Avoid interactive wandb login prompts by default.
    wandb_mode = conf.get("wandb_mode")
    if wandb_mode is None:
        wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"
    if conf.get("wandb_project"):
        wandb.init(project=conf["wandb_project"], name=conf["wandb_run_name"], config=conf, mode=wandb_mode)
    start_iter = 0
    if conf["resume"]:
        start_iter = load_checkpoint(conf["resume"], model, optimizer)
        print(f"load checkpoint at iteration {start_iter} from file {conf['resume']}")

    ckpt_dir = _resolve_path(conf["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for iter in range(start_iter, conf["max_iter"]):
        lr = lr_cosine_schedule(
            iter, conf["lr_max"], conf["lr_min"], conf["warmup_iters"], conf["cosine_iters"]
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        xb, yb = get_batch(train_data, batch_size, context_length, device)
        logits = model(xb)
        loss = cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), conf["grad_clip"])
        optimizer.step()

        if conf["eval_iter"] and iter % conf["eval_iter"] == 0:
            losses = estimate_loss(model, train_data, val_data, batch_size, context_length, device, conf["eval_runs"])
            print(f"iter = {iter}, train_loss = {losses['train']}, val_loss = {losses['val']}")
            if wandb.run is not None:
                wandb.log({"iter": iter, "train_loss": losses["train"], "val_loss": losses["val"]})

        if conf["ckpt_iter"] and iter % conf["ckpt_iter"] == 0:
            ckpt_path = ckpt_dir / f"iter_{iter:06d}.pt"
            save_checkpoint(model, optimizer, iter, ckpt_path)
            print(f"save checkpoint at iteration {iter} to file {ckpt_path}")

    final_ckpt = ckpt_dir / "final.pt"
    save_checkpoint(model, optimizer, conf["max_iter"], final_ckpt)
    print(f"save final checkpoint to file {final_ckpt}")

    losses = estimate_loss(model, train_data, val_data, batch_size, context_length, device, conf["eval_runs"])
    print(f"final train_loss = {losses['train']}, val_loss = {losses['val']}")
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
