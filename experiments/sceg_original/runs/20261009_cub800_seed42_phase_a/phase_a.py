"""Read-only source/data checks; outputs are exclusive to this Phase A run."""
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
from datasets import make_dataset
from datasets.samplers import TrainingEpisodeSampler, EvaluationEpisodeSampler
from utils.config import load_config
from utils.experiment import environment_info, source_manifest

RUN = Path(__file__).resolve().parent if len(sys.argv) == 1 else ROOT / "experiments/sceg_original/runs" / sys.argv[1]
if len(sys.argv) > 1:
    RUN.mkdir(exist_ok=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (RUN / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    start = time.monotonic()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(RUN / "phase_a.log", mode="x"),
                                  logging.StreamHandler()])
    config_path = ROOT / "experiments/sceg_original/configs/cub.yaml"
    config = load_config(config_path)
    assert config["lambda"] == config["loss"]["lamda"] == 38
    assert config["data"]["workers"] == 4
    expected = load_config(ROOT / "experiments/sceg_original/config.yaml")
    for section in ("model", "train", "loss", "runtime", "verification"):
        assert config[section] == expected[section], section
    assert {k: v for k, v in config["data"].items() if k not in ("workers", "pin_memory")} == {
        k: v for k, v in expected["data"].items() if k != "workers"}
    assert all(config["eval"][k] == v for k, v in expected["eval"].items())
    for name in ("config.yaml", "config_resolved.yaml"):
        with (RUN / name).open("x") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    save("environment.json", environment_info())
    save("protected_source_manifest.json", source_manifest(ROOT))
    protected = {}
    for family in ("sceg_original", "sceg_debug", "sceg_preflight"):
        for path in (ROOT / "experiments" / family).rglob("*"):
            if path.is_file() and RUN not in path.parents and "__pycache__" not in path.parts:
                protected[str(path.relative_to(ROOT))] = digest(path)
    save("existing_artifact_hashes.json", protected)
    commit = subprocess.check_output(["rtk", "proxy", "git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (RUN / "git_commit.txt").write_text(commit + "\n")
    (RUN / "command.txt").write_text(" ".join(sys.orig_argv) + "\n")
    official = Path("/tmp/sceg-official-reference")
    official_commit = subprocess.check_output(["rtk", "proxy", "git", "rev-parse", "HEAD"], cwd=official, text=True).strip()
    assert official_commit == config["official_commit"]
    dirty = subprocess.check_output(["rtk", "proxy", "git", "status", "--porcelain"], cwd=official, text=True)
    assert not dirty.strip(), "Official reference checkout is dirty"
    source = yaml.safe_load((ROOT / "docs/sources/manifest.yaml").read_text())
    frn = ROOT / "data/provenance/frn_init_CUB_fewshot.py"
    if not frn.exists():
        candidates = [p for p in (ROOT / "data/provenance").rglob("*.py")
                      if digest(p) == source["frn_cub_split"]["source_sha256"]]
        if len(candidates) == 1:
            frn = candidates[0]
        else:
            url = "https://raw.githubusercontent.com/Tsingularity/FRN/" + source["frn_cub_split"]["commit"] + "/data/init_CUB_fewshot.py"
            published = urllib.request.urlopen(url, timeout=30).read()
            assert hashlib.sha256(published).hexdigest() == source["frn_cub_split"]["source_sha256"]
            frn = url
    if isinstance(frn, Path):
        assert digest(frn) == source["frn_cub_split"]["source_sha256"]
    save("source_provenance.json", {"official_repository": source["official_repository"],
        "frn_split": source["frn_cub_split"], "frn_local_source": str(frn),
        "official_recipe_sha256": digest(official / "scripts/run.sh"),
        "config_sha256": digest(config_path), "local_commit": commit})
    data_root = ROOT / config["data"]["root"]
    copied = json.loads((ROOT / "data/provenance/copied_datasets.json").read_text())
    record = next(v for v in copied.values() if Path(v["destination"]).resolve() == data_root)
    pre = json.loads((ROOT / "data/provenance/pre_resized.json").read_text())["CUB_fewshot_raw"]
    files, counts, modes, dimensions = {}, {}, Counter(), Counter()
    class_lists = {}
    resize_crop = transforms.Compose([transforms.Resize(92), transforms.CenterCrop(84)])
    for split, class_count, image_count in (("train", 100, 5903), ("val", 50, 2927), ("test", 50, 2958)):
        classes = sorted(p.name for p in (data_root / split).iterdir() if p.is_dir())
        ids = [int(name.split(".")[0]) for name in classes]
        assert ids == [i for i in range(1, 201) if
            (i % 2 == 0 if split == "train" else i % 4 == (1 if split == "val" else 3))]
        class_lists[split] = classes
        paths = sorted(p for p in (data_root / split).rglob("*") if p.is_file())
        assert len(classes) == class_count and len(paths) == image_count
        for path in paths:
            assert not path.is_symlink()
            relative = str(path.relative_to(data_root))
            files[relative] = digest(path)
            assert files[relative] == record["files"][relative], relative
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                modes[image.mode] += 1
                rgb = image.convert("RGB")
                assert rgb.mode == "RGB"
                dimensions[str(rgb.size)] += 1
                if split != "train":
                    target_relative = path.relative_to(data_root / split).with_suffix(".png")
                    target = data_root / (split + "_pre") / target_relative
                    assert not target.is_symlink()
                    assert digest(target) == pre[split + "_pre"]["files"][str(target_relative)]
                    with Image.open(target) as saved:
                        saved.load()
                        assert saved.mode == "RGB" and saved.size == (84, 84)
                        assert np.array_equal(np.asarray(resize_crop(rgb)), np.asarray(saved))
        pre_count = None if split == "train" else sum(p.is_file() for p in (data_root / (split + "_pre")).rglob("*"))
        assert pre_count is None or pre_count == image_count
        counts[split] = {"classes": len(classes), "images": len(paths), "pre_resized_images": pre_count}
        logging.info("split=%s classes=%d images=%d SHA256/decode/RGB/crop checks passed", split, len(classes), len(paths))
    tree_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    assert tree_hash == record["tree_sha256"]
    save("image_hashes.json", files)
    save("data_manifest.json", {"dataset": "CUB", "root": str(data_root),
        "class_split_source": source["frn_cub_split"], "train_classes": class_lists["train"],
        "val_classes": class_lists["val"], "test_classes": class_lists["test"],
        "image_count": sum(c["images"] for c in counts.values()), "split_counts": counts,
        "hash": {"algorithm": "sha256", "tree_sha256": tree_hash,
                 "per_image_manifest": "image_hashes.json", "per_image_manifest_sha256": digest(RUN / "image_hashes.json")},
        "raw_image_modes": dict(modes), "all_images_decoded": True, "loader_rgb_conversion": True,
        "all_pre_resized_pixels_equal": True, "resize_crop": "PIL RGB -> Resize(short edge92) -> CenterCrop84 -> PNG",
        "training_transform": "RGB -> RandomResizedCrop84 -> ColorJitter(.4,.4,.4) -> RandomHorizontalFlip(.5) -> ToTensor -> ImageNetNormalize"})
    np.random.seed(42)
    dataset = make_dataset(config, "train")
    sampler = TrainingEpisodeSampler(dataset, 10, (5, 15))
    episode_indices = list(sampler)
    used = [i for batch in episode_indices for i in batch]
    assert len(used) == len(set(used))
    for batch in episode_indices:
        labels = np.array([dataset.targets[i] for i in batch])
        support, query = labels[:50].reshape(10, 5), labels[50:].reshape(10, 15)
        assert (support == support[:, :1]).all() and (query == support[:, :1]).all()
        assert len(set(support[:, 0])) == 10
    for split in ("train", "val", "test"):
        data = make_dataset(config, split)
        image, _ = data[0]
        assert image.shape == (3, 84, 84) and torch.isfinite(image).all()
    heldout = make_dataset(config, "test")
    for way in (5, 50):
        for shot in (1, 5):
            episode = next(iter(EvaluationEpisodeSampler(heldout, way, shot, 1, 16)))
            assert len(episode) == way * (shot + 16) and len(episode) == len(set(episode))
    save("phase_a_checks.json", {"status": "passed", "run_id": RUN.name,
        "train_episode_count_seed42_sampler_only": len(episode_indices),
        "train_unique_images_consumed": len(used), "full_data_manifest_verified": True,
        "recipe_matches_prior_confirmed_config": True, "workers": 4,
        "formal_training_started": False, "elapsed_seconds": time.monotonic() - start})
    logging.info("Phase A passed; episodes=%d elapsed=%.1fs", len(episode_indices), time.monotonic() - start)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        save("failure.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        logging.exception("Phase A failed")
        raise
