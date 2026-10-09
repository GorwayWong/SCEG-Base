"""Official ImageFolder loading; class IDs come from supplied directories, never resplit."""
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from .transforms import image_transform
from .samplers import TrainingEpisodeSampler, EvaluationEpisodeSampler
from utils.reproducibility import seed_worker


def make_dataset(config, split):
    if split not in {"train", "val", "test"}:
        raise ValueError("Unknown split")
    training = split == "train"
    pre_resized = config["data"].get("pre_resized", False) and not training
    path = Path(config["data"]["root"])/(split+"_pre" if pre_resized else split)
    if not path.is_dir():
        raise FileNotFoundError(f"Missing official prepared split: {path}")
    return ImageFolder(path, transform=image_transform(config, training, pre_resized))


def make_loader(config, split, way=None, shot=None, episodes=None):
    dataset = make_dataset(config, split)
    if split == "train":
        sampler = TrainingEpisodeSampler(dataset, config["model"]["train_way"],
                                          (config["model"]["train_shot"], config["model"]["train_query"]))
    else:
        sampler = EvaluationEpisodeSampler(dataset, way, shot, episodes, query=config["eval"]["query"])
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=config["data"]["workers"],
                        pin_memory=False, worker_init_fn=seed_worker)
    return loader
