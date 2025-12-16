"""
1. Ability to configure and control the various model and optimizer hyperparameters.
2. Memory-efficient loading of training and validation large datasets with np.memmap.
3. Serializing checkpoints to a user-provided path.
4. Periodically logging training and validation performance.
   (e.g., to console and/or an external service like Weights and Biases)
"""

import torch
import json
from pathlib import Path
import argparse
import wandb

from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import *
from cs336_basics.utils import *

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"


def get_config() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=CONFIG_DIR / "config.json")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_memmap(path: Path, dtype: str) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(f"Memmap file not found {path}")
    return np.memmap(path, dtype, "r")


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
    # 1. read config and load data
    config = get_config()
    train_data = load_memmap(config["train_data"], config["dtype"])
    val_data = load_memmap(config["val_data"], config["dtype"])
    batch_size = config["batch_size"]
    context_length = config["context_length"]
    device = config["device"]
    dtype = config["dtype"]

    # 2. init the model and optimizer
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=device,
        dtype=dtype,
    )
    optimizer = AdamW(
        params=model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["eps"],
    )

    # 3. start training loop
    wandb.init(project=config["wandb_project"], name=config["wandb_run_name"], config=config)
    start_iter = 0
    if config["resume"]:
        start_iter = load_checkpoint(config["resume"], model, optimizer)
        print(f"load checkpoint at iteration {start_iter} from file {config['resume']}")

    ckpt_dir = Path(config["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for iter in range(start_iter, config["max_iter"]):
        lr = lr_cosine_schedule(
            iter, config["lr_max"], config["lr_min"], config["warmup_iters"], config["cosine_iters"]
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        xb, yb = get_batch(train_data, batch_size, context_length, device)
        logits = model(xb)
        loss = cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), config["grad_clip"])
        optimizer.step()

        if config["eval_iter"] and iter % config["eval_iter"] == 0:
            losses = estimate_loss(model, train_data, val_data, batch_size, context_length, device, config["eval_runs"])
            print(f"iter = {iter}, train_loss = {losses['train']}, val_loss = {losses['val']}")
            wandb.log({"iter": iter, "train_loss": losses["train"], "val_loss": losses["val"]})

        if config["ckpt_iter"] and iter % config["ckpt_iter"] == 0:
            ckpt_path = ckpt_dir / f"iter_{iter:06d}.pt"
            save_checkpoint(model, optimizer, iter, ckpt_path)
            print(f"save checkpoint at iteration {iter} to file {ckpt_path}")

    final_ckpt = ckpt_path / "final.pt"
    save_checkpoint(model, optimizer, config["max_iter"], final_ckpt)
    print(f"save final checkpoint to file {final_ckpt}")

    losses = estimate_loss(model, train_data, val_data, batch_size, context_length, device, config["eval_runs"])
    print(f"final train_loss = {losses['train']}, val_loss = {losses['val']}")
    wandb.finish()


if __name__ == "__main__":
    main()
