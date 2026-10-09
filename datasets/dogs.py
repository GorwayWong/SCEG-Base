"""Dogs class-disjoint folders: supplement §7.1 (70/20/30). No guessed splits."""
from .folder import make_dataset


def load_dogs(config, split):
    assert config["model"]["dataset"] == "stanford_dog"
    return make_dataset(config, split)
