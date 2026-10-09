"""CUB class-disjoint folders: supplement §7.1 (100/50/50), supplied FRN-style split."""
from .folder import make_dataset


def load_cub(config, split):
    assert config["model"]["dataset"] == "cub_raw"
    return make_dataset(config, split)
