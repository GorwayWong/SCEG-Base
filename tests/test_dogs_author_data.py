"""Author-source and data-only preparation checks; no model/experiment smoke."""
import io
import json
from pathlib import Path
import shutil
import tarfile

from PIL import Image
import pytest

from experiments.sceg_dogs_author.prepare import COUNTS, parse_classes, prepare_data, verify_data
from tools.prepare_data import sha256_file


def author_fixture(path):
    breeds = [f"n{index:08d}-fixture" for index in range(120)]
    groups = {"train": breeds[:70], "val": breeds[70:90], "test": breeds[90:]}
    text = "Explicit synthetic data-preparation fixture; never real Dogs evidence\n"
    for part, ids in groups.items():
        text += f"[{part}] ({len(ids)} classes)\n"+"\n".join(ids)+"\n"
    path.write_text(text)
    return groups


def test_author_class_count_and_overlap_validation(tmp_path):
    path = tmp_path/"classes.txt"
    groups = author_fixture(path)
    assert parse_classes(path) == groups
    path.write_text(path.read_text().replace(groups["val"][0], groups["train"][0]))
    with pytest.raises(ValueError, match="overlapping"):
        parse_classes(path)


def test_cars_original_ids_are_one_based_and_complete(tmp_path):
    path = tmp_path/"cars.txt"
    groups = {"train": list(range(1, 131)), "val": list(range(131, 148)), "test": list(range(148, 197))}
    path.write_text("Fixture-only IDs\n"+"".join(f"[{part}] ({len(ids)} classes)\n"+"\n".join(map(str, ids))+"\n"
                                              for part, ids in groups.items()))
    assert parse_classes(path, {"train": 130, "val": 17, "test": 49}, numeric=True) == groups
    path.write_text(path.read_text().replace("\n196\n", "\n0\n"))
    with pytest.raises(ValueError, match="1..196"):
        parse_classes(path, {"train": 130, "val": 17, "test": 49}, numeric=True)


def test_preparation_reuse_is_independent_of_deleted_donor(tmp_path):
    source = tmp_path/"sources"
    source.mkdir()
    author = source/"StanfordDogs_fewshot_splits.txt"
    groups = author_fixture(author)
    (source/"source.json").write_text(json.dumps({"source_kind": "synthetic_fixture_only",
                                                "files": {"dogs": {"sha256": sha256_file(author)}}}))
    image = io.BytesIO()
    Image.new("RGB", (100, 125), (25, 50, 75)).save(image, format="JPEG")
    content = image.getvalue()
    archive = tmp_path/"images.tar"
    donor = tmp_path/"donor"
    with tarfile.open(archive, "w") as stream:
        for breeds in groups.values():
            for breed in breeds:
                info = tarfile.TarInfo(f"Images/{breed}/image.jpg")
                info.size = len(content)
                stream.addfile(info, io.BytesIO(content))
                # Donor groups intentionally all train; author val/test need new PNGs.
                path = donor/"train"/breed/"image.jpg"
                path.parent.mkdir(parents=True)
                path.write_bytes(content)
    audit = tmp_path/"audit.json"
    audit.write_text(json.dumps({"dogs": {"bytes": archive.stat().st_size, "sha256": sha256_file(archive)}}))
    root, prep = tmp_path/"ready", tmp_path/"prep"
    result = prepare_data(root, prep, source, archive, audit, donor, expected_images=120)
    assert result["data_ready"] and result["training_started"] is False
    assert result["reuse"]["raw_hardlink"] == 120
    assert result["reuse"]["pre_generated"] == 50
    assert not any(path.is_symlink() for path in root.rglob("*"))
    shutil.rmtree(donor)
    assert verify_data(root, prep, result["reuse"]["new_pre_classes"])["full_inventory_checked"]
    assert sha256_file(root/"train"/groups["train"][0]/"image.jpg") == __import__("hashlib").sha256(content).hexdigest()
