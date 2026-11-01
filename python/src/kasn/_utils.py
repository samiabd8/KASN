"""Small internal helpers: devices, seeding and serialisation hygiene."""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Optional, Union

import numpy as np
import torch


def resolve_device(device: Union[str, torch.device, None]) -> torch.device:
    """``'auto'`` (or ``None``) picks CUDA when available, otherwise CPU."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@contextlib.contextmanager
def seeded(random_state: Optional[int],
           device: torch.device) -> Iterator[None]:
    """Seed torch inside a forked RNG so the caller's global state is untouched.

    Every candidate in a hyperparameter search therefore starts from the same
    initialisation, and repeated fits do not consume each other's draws.
    """
    if random_state is None:
        yield
        return
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None
                   else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(random_state))
        yield


def plain(obj: Any) -> Any:
    """Recursively convert numpy scalars and arrays to built-in Python types.

    ``torch.load(weights_only=True)`` accepts only built-in containers and
    tensors, so everything saved alongside the state dict passes through here.
    """
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(plain(v) for v in obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.device):
        return str(obj)
    return obj
