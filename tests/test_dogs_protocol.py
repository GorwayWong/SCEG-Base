"""New Dogs provenance gates; fixtures are explicitly not author splits."""
import json
import pytest
from experiments.sceg_dogs_baseline.prepare import trusted_split
from tools.prepare_data import sha256_file


def test_counts_without_class_id_source_are_blocked(tmp_path):
    path = tmp_path/"split.json"
    path.write_text(json.dumps({"status": "blocked", "expected_counts": [70, 20, 30]}))
    with pytest.raises(RuntimeError, match="BLOCKED"):
        trusted_split(path, [])


@pytest.mark.parametrize("problem", ["overlap", "hash", "missing_source"])
def test_untrusted_or_overlapping_ids_are_rejected(tmp_path, problem):
    evidence = tmp_path/"fixture_only.txt"
    evidence.write_text("Synthetic class identities, never author protocol")
    ids = [f"synthetic_{i:03d}" for i in range(120)]
    source = {"url": "fixture://technical-test", "revision": "fixture-v1",
              "artifact_path": str(evidence), "artifact_sha256": sha256_file(evidence),
              "mapping_method": "synthetic unit test"}
    classes = {"train": ids[:70], "val": ids[70:90], "test": ids[90:]}
    if problem == "overlap":
        classes["test"][0] = ids[0]
    elif problem == "hash":
        source["artifact_sha256"] = "0"*64
    else:
        source.pop("artifact_path")
    path = tmp_path/"split.json"
    path.write_text(json.dumps({"status": "verified", "source": source, "classes": classes}))
    with pytest.raises(ValueError):
        trusted_split(path, ids)


def test_preparation_and_inventory_guard(tmp_path, monkeypatch):
    import io
    import tarfile
    from PIL import Image
    from utils.config import load_config
    from experiments.sceg_dogs_baseline import prepare as adapter
    config = load_config(adapter.BASE/"configs/dogs.yaml")
    evidence = tmp_path/"synthetic_evidence.txt"
    evidence.write_text("Explicit synthetic protocol fixture, not Dogs author split")
    breeds = [f"synthetic_{i:03d}" for i in range(120)]
    split_file = tmp_path/"split.json"
    split_file.write_text(json.dumps({"status": "verified", "source": {
        "url": "fixture://technical-test", "revision": "unit-test", "artifact_path": str(evidence),
        "artifact_sha256": sha256_file(evidence), "mapping_method": "synthetic test only"},
        "classes": {"train": breeds[:70], "val": breeds[70:90], "test": breeds[90:]}}))
    image = io.BytesIO()
    Image.new("RGB", (30, 25), (32, 64, 128)).save(image, format="PNG")
    content = image.getvalue()
    archive = tmp_path/"images.tar"
    with tarfile.open(archive, "w") as stream:
        for breed in breeds:
            for index in range(21):
                info = tarfile.TarInfo(f"Images/{breed}/{index:03d}.png")
                info.size = len(content)
                stream.addfile(info, io.BytesIO(content))
    old_manifest = tmp_path/"archive_manifest.json"
    old_manifest.write_text(json.dumps({"dogs": {"bytes": archive.stat().st_size,
                                                 "sha256": sha256_file(archive)}}))
    config["data"]["root"] = str(tmp_path/"prepared")
    config["data_protocol"] = {"archive": str(archive), "existing_archive_manifest": str(old_manifest),
        "split_manifest": str(split_file), "prepared_manifest": str(tmp_path/"prepared.json")}
    (tmp_path/"data_protocol").mkdir()
    (tmp_path/"data_protocol/raw_inventory.json").write_text(json.dumps({"class_images": breeds}))
    monkeypatch.setattr(adapter, "BASE", tmp_path)
    record = adapter.prepare(config)
    assert {key: len(value["class_to_idx"]) for key, value in record["classes"].items()} == adapter.COUNTS
    assert adapter.check_prepared(config)["pre_pixels_verified"]
    path = next((tmp_path/"prepared/train").rglob("*.png"))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="state changed"):
        adapter.check_prepared(config)


def test_suite_reuses_five_way_and_same_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.sceg_dogs_baseline import run as entry
    (tmp_path/"checkpoint.pth").write_bytes(b"synthetic checkpoint fixture")
    seen = []
    monkeypatch.setattr(entry.torch, "load", lambda *args, **kwargs: {
        "best_epoch": 20, "model_state_dict": {}})
    def evaluate(model, config, split, way, shot, episodes):
        seen.append((way, shot))
        return {"way": way, "shot": shot, "query": 16, "episodes": episodes,
                "accuracy": 0.0, "confidence_interval": 0.0}
    monkeypatch.setattr(entry, "evaluate_episodes", evaluate)
    for shot in (1, 5):
        (tmp_path/f"test_5way_{shot}shot.json").write_text(json.dumps(
            {"way": 5, "shot": shot, "query": 16, "episodes": 2000, "accuracy": 0.0}))
    metrics = {}
    recorder = SimpleNamespace(path=tmp_path, update_metrics=lambda **values: metrics.update(values))
    model = SimpleNamespace(load_state_dict=lambda state, strict: None)
    entry.complete_suite(model, {"eval": {"test_episodes": 2000}}, recorder)
    assert seen == [(30, 1), (30, 5)]
    assert len({item["checkpoint_sha256"] for item in metrics["evaluation"]}) == 1
    assert metrics["full_evaluation_completed"] is True
    assert not (tmp_path/"checkpoint_30way.pth").exists()
