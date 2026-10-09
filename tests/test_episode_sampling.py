"""Validate data semantics against the official sampler as an isolated optional oracle."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from datasets.samplers import TrainingEpisodeSampler, EvaluationEpisodeSampler


def sample_dataset():
    samples = [(f"class{category}/image{index}.jpg", category)
               for category in range(6) for index in range(25+category)]
    return SimpleNamespace(samples=samples, imgs=samples)


def test_training_nonoverlap_support_order():
    dataset = sample_dataset()
    seen = set()
    for episode in TrainingEpisodeSampler(dataset, way=3, shots=(1, 2)):
        assert len(episode) == 9
        assert not seen.intersection(episode)
        seen.update(episode)
        support_labels = [dataset.samples[index][1] for index in episode[:3]]
        query_labels = [dataset.samples[index][1] for index in episode[3:]]
        assert query_labels == [category for category in support_labels for _ in range(2)]


def test_evaluation_ignores_query_and_uses_disjoint_samples():
    dataset = sample_dataset()
    sampler = EvaluationEpisodeSampler(dataset, way=3, shot=1, episodes=4, query=15)
    assert sampler.query == 16
    episodes = list(sampler)
    assert len(episodes) == 4
    for episode in episodes:
        assert len(episode) == 3*(1+16)
        assert len(set(episode)) == len(episode)


def test_official_sampler_oracle():
    path = Path("/tmp/sceg-official-reference/datasets/samplers.py")
    if not path.exists():
        pytest.skip("External official checkout not available")
    spec = importlib.util.spec_from_file_location("official_samplers", path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    dataset = sample_dataset()
    for replica, reference in (
        (TrainingEpisodeSampler(dataset, 3, (1, 2)), official.meta_batchsampler(dataset, 3, (1, 2))),
        (EvaluationEpisodeSampler(dataset, 3, 1, 4, 15), official.random_sampler(dataset, 3, 1, 15, 4)),
    ):
        np.random.seed(42); expected = list(reference)
        np.random.seed(42); actual = list(replica)
        assert actual == expected
