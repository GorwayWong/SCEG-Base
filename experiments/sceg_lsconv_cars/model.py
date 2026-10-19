"""Only migration: SC.self_s(F_s) -> official LSNet LSConv(F_s)."""
import torch
from models import SCEG
from utils.config import ModelConfig


def build_model(config, enabled=True):
    model = SCEG(ModelConfig.from_dict(config["model"]))
    if not enabled:
        return model
    # Construct on CPU after SCEG's initialization pass. Preserve subsequent RNG.
    with torch.random.fork_rng(devices=[]):
        from .lsconv import LSConv
        model.SC.self_s = LSConv(dim=320)
    return model
