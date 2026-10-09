"""Flowers class-disjoint folders: supplement §7.1 (51/26/25); not Oxford's image split."""
from .folder import make_dataset


def load_flowers(config, split):
    assert config["model"]["dataset"] == "flower"
    return make_dataset(config, split)
