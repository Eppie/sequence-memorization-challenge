# ── Device selection ─────────────────────────────────────────────────────────
# Central place to decide which torch device the code runs on, so the same
# scripts work on a CUDA pod, a CPU-only box, or a Mac laptop.
#
# Priority:
#   1. The MTM_DEVICE environment variable, if set (e.g. "cuda", "cpu", "cuda:0").
#      The parallel runner sets this per-process so you can pin jobs to devices.
#   2. "cuda" if a GPU is available.
#   3. "cpu" otherwise.

import os
import torch


def get_device() -> str:
    """Return the torch device string this process should use."""
    dev = os.environ.get("MTM_DEVICE")
    if dev:
        return dev
    return "cuda" if torch.cuda.is_available() else "cpu"


def setup_default_device() -> str:
    """Set the global default device and return it.

    Call this once, early (before any tensors/models are created). ``models.py``
    calls it on import, so importing models is enough for most scripts.
    """
    device = get_device()
    torch.set_default_device(device)
    return device
