"""Typed model settings and immutable run configuration loading."""
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class ModelConfig:
    dataset: str
    input_channels: int
    image_size: int
    stage_channels: tuple[int, ...]
    feature_dim: int
    base_classes: int
    train_way: int
    train_shot: int
    train_query: int
    transform_alpha: float
    gate_hidden_channels: int
    stage3_to_stage2_datasets: tuple[str, ...]
    dropout: float
    dropblock_size: int
    dropblock_ramp_batches: int
    negative_slope: float
    batchnorm_eps: float
    batchnorm_momentum: float
    projection_bn_momentum: float
    episode_norm_eps: float
    global_norm_eps: float
    convolution_kernel: int
    pool_stride: int

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for field in ("stage_channels", "stage3_to_stage2_datasets"):
            values[field] = tuple(values[field])
        config = cls(**values)
        if len(config.stage_channels) != 4 or min(config.stage_channels) < 1:
            raise ValueError("ResNet12 requires four positive stage channels")
        if config.dataset not in {"cub_raw", "stanford_dog", "stanford_car", "flower"}:
            raise ValueError("Unknown official dataset branch")
        if not 0 <= config.transform_alpha <= 1 or not 0 <= config.dropout < 1:
            raise ValueError("Invalid transform/dropout configuration")
        return config


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    ModelConfig.from_dict(config["model"])
    return config
