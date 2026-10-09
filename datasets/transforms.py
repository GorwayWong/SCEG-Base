"""Official datasets/transform_manager.py:18–60; all hyperparameters in config."""
from torchvision import transforms


def image_transform(config, training, pre_resized=False):
    data = config["data"]
    normalize = [transforms.ToTensor(), transforms.Normalize(data["mean"], data["std"])]
    if training:
        if pre_resized:
            raise ValueError("Official training does not use pre-resized evaluation folders")
        return transforms.Compose([
            transforms.RandomResizedCrop(data["image_size"], scale=tuple(data["crop_scale"]),
                                          ratio=tuple(data["crop_ratio"])),
            transforms.ColorJitter(brightness=data["jitter"], contrast=data["jitter"], saturation=data["jitter"]),
            transforms.RandomHorizontalFlip(data["flip_probability"]), *normalize,
        ])
    if pre_resized:
        return transforms.Compose(normalize)
    return transforms.Compose([transforms.Resize(data["eval_resize"]),
                               transforms.CenterCrop(data["image_size"]), *normalize])
