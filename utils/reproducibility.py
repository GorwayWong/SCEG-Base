"""Control Python, NumPy, torch, CUDA, workers, and backend randomness."""
import os
import random
import numpy as np
import torch


def seed_everything(seed, deterministic=True, allow_tf32=False, warn_only=False):
    # CUBLAS setting must precede the first CUDA computation.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
