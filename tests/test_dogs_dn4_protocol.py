"""CSV provenance and isolated records; no model or synthetic training repeat."""
import copy
import csv
import json

import pytest
from experiments.sceg_dogs_baseline.prepare import csv_rows, trusted_split
from tools.prepare_data import sha256_file


def source_fixture(tmp_path):
    breeds = [f"fixture_{i:03}" for i in range(120)]
    classes = {"train": breeds[:70], "val": breeds[70:90], "test": breeds[90:]}
    source = {"url": "fixture://unit-test", "revision": "fixture-v1",
              "mapping_method": "Synthetic CSV unit fixture", "artifacts": {}}
    for part, ids in classes.items():
        path = tmp_path/(part+".csv")
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["filename", "label"])
            writer.writerows(("image.jpg", label) for label in ids)
        source["artifacts"][part] = {"url": "fixture://"+part, "path": str(path), "sha256": sha256_file(path)}
    return breeds, classes, source


def test_csv_source_and_exact_class_identities(tmp_path):
    breeds, classes, source = source_fixture(tmp_path)
    path = tmp_path/"split.json"
    manifest = {"status": "verified", "source": source, "classes": classes}
    path.write_text(json.dumps(manifest))
    assert trusted_split(path, breeds)["classes"] == classes
    manifest["classes"]["train"][0], manifest["classes"]["val"][0] = (
        manifest["classes"]["val"][0], manifest["classes"]["train"][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="CSV class identities"):
        trusted_split(path, breeds)


@pytest.mark.parametrize("problem", ["duplicate", "path", "hash"])
def test_invalid_csv_rejected(tmp_path, problem):
    _, _, source = source_fixture(tmp_path)
    artifact = source["artifacts"]["train"]
    with open(artifact["path"], "a") as stream:
        stream.write("image.jpg,fixture_000\n" if problem != "path" else "../image.jpg,fixture_000\n")
    if problem != "hash":
        artifact["sha256"] = sha256_file(artifact["path"])
    with pytest.raises(ValueError, match={"duplicate": "Duplicate", "path": "Invalid", "hash": "hash"}[problem]):
        csv_rows(source)


def test_dn4_entry_sources_and_results_are_isolated(tmp_path, monkeypatch):
    from experiments.sceg_dogs_baseline.run import DogsRun
    from experiments.sceg_dogs_dn4.prepare import BASE
    from utils.config import load_config
    import utils.experiment as recording
    config = load_config(BASE/"configs/dogs.yaml")
    monkeypatch.setattr(recording, "__file__", str(tmp_path/"utils/experiment.py"))
    monkeypatch.setattr(recording, "environment_info", lambda: {"fixture": True})
    base = tmp_path/"experiments/sceg_dogs_dn4"
    (base/"configs").mkdir(parents=True)
    (base/"configs/dogs.yaml").write_text("fixture: true\n")
    (base/"run.py").write_text("# synthetic source fixture\n")
    (tmp_path/"tests").mkdir()
    (tmp_path/"tests/test_dogs_dn4_protocol.py").write_text("# synthetic test fixture\n")
    run = DogsRun(copy.deepcopy(config), "fixture")
    run.publish_result()
    snapshot = (run.path/"experiment_source.patch").read_text()
    assert "sceg_dogs_dn4/run.py" in snapshot and "sceg_dogs_baseline" not in snapshot
    assert (base/"result.json").exists()
    assert not (tmp_path/"experiments/sceg_original/result.json").exists()
    assert not (tmp_path/"experiments/sceg_dogs_baseline/result.json").exists()


def test_evaluation_metadata_preserves_random_stream():
    import random
    import numpy as np
    import torch
    from experiments.sceg_dogs_baseline.run import evaluation_rng, restore_evaluation_rng
    state = json.loads(json.dumps(evaluation_rng()))
    expected = (random.random(), np.random.rand(3), torch.rand(3))
    restore_evaluation_rng(state)
    actual = (random.random(), np.random.rand(3), torch.rand(3))
    assert actual[0] == expected[0]
    assert np.array_equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])
