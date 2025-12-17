import torch
from torch import Tensor
import numpy.typing as npt
import numpy as np
import os
import typing
from pathlib import Path
import argparse
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
CONF_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    data_length = len(dataset)
    ix = np.random.randint(low=0, high=data_length - context_length, size=batch_size)

    x = np.stack([dataset[i : i + context_length] for i in ix])
    y = np.stack([dataset[i + 1 : i + 1 + context_length] for i in ix])
    x = torch.from_numpy(x).long().to(device)
    y = torch.from_numpy(y).long().to(device)
    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    obj = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]


def get_conf(conf: str | Path) -> dict:
    with open(conf, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def resolve_path(p: str | os.PathLike) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def torch_dtype_from_str(dtype: str | None) -> torch.dtype | None:
    if dtype is None:
        return None
    if isinstance(dtype, str):
        # Accept common strings like "float32", "bfloat16", etc.
        t = getattr(torch, dtype, None)
        return t if isinstance(t, torch.dtype) else None
    return None


def bytes_per_elem(dtype: torch.dtype) -> int:
    # Minimal mapping for common dtypes used in this repo.
    if dtype in (torch.float16, torch.bfloat16, torch.int16, torch.uint16):
        return 2
    if dtype in (torch.float32, torch.int32, torch.uint32):
        return 4
    if dtype in (torch.float64, torch.int64, torch.uint64):
        return 8
    # Fallback for unexpected dtypes.
    try:
        return torch.tensor([], dtype=dtype).element_size()
    except Exception:
        return 4


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: device={device!r} requested but CUDA not available; falling back to CPU.")
        return "cpu"
    return device