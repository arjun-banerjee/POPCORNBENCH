"""Utilities for optional multi-process (torchrun) reference problems in ``distributed/``."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed_run() -> bool:
    return (
        dist.is_available()
        and "RANK" in os.environ
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )


def maybe_init_process_group() -> None:
    if not is_distributed_run():
        return
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def get_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def default_device() -> torch.device:
    if torch.cuda.is_available() and is_distributed_run():
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
