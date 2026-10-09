"""Cars class-disjoint folders: supplement §7.1 (130/17/49); preserve supplied class IDs."""
from .folder import make_dataset


def load_cars(config, split):
    assert config["model"]["dataset"] == "stanford_car"
    return make_dataset(config, split)
