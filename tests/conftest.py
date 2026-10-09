from pathlib import Path
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import load_config, ModelConfig
from utils.reproducibility import seed_everything


@pytest.fixture
def config():
    return load_config(ROOT / "configs/sceg_original.yaml")


@pytest.fixture
def model_config(config):
    return ModelConfig.from_dict(config["model"])


@pytest.fixture(autouse=True)
def deterministic_runtime(config):
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    seed_everything(config["seed"], deterministic=config["runtime"]["deterministic"],
                    allow_tf32=config["runtime"]["allow_tf32"],
                    warn_only=config["runtime"]["deterministic_warn_only"])


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
