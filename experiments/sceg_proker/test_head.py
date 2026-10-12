"""Only the necessary invariants of this independent inference head."""
from pathlib import Path
import pytest
import torch
from experiments.sceg_proker.head import VisualEpisodeHead, kernel_features, rbf_kernel, residual_coefficients
from utils.config import load_config


def episode():
    torch.manual_seed(123)
    z = torch.randn(95, 3, 640)
    labels = torch.arange(5).repeat_interleave(3)
    config = load_config(Path(__file__).resolve().parents[1]/"sceg_original/configs/cub.yaml")
    head = VisualEpisodeHead(config["model"]).fit(z[:15], labels, 5)
    return head, z, labels


def test_author_core_same_inputs():
    head, z, _ = episode()
    us, uq = kernel_features(z[:15]), kernel_features(z[15:])
    beta, lmbda = 5., .1
    # Exact pinned trainer formulas, including float casts and the identity scale.
    kss = (-beta*(1-us.float()@us.float().T)).exp()
    r = head.targets-head.baseline(z[:15])
    author_a = torch.linalg.solve(1/lmbda*kss+torch.eye(15), r)
    author_scores = head.baseline(z[15:])+(-beta*(1-uq.float()@us.float().T)).exp()@author_a
    torch.testing.assert_close(residual_coefficients(rbf_kernel(us, us, beta), r, lmbda), author_a)
    torch.testing.assert_close(head.proker(z[15:], beta, lmbda, 1.), author_scores)
    conventional_a = torch.linalg.solve(kss+lmbda*torch.eye(15), r)
    torch.testing.assert_close(author_a, lmbda*conventional_a)


def test_zero_residual_original_scores_and_predictions():
    head, z, _ = episode()
    original = [head.classifier(z[:, s], 5, 3) for s in range(3)]
    original = (original[1]+original[2]+original[0])/3
    actual = head.proker(z[15:], 5., 1., 0.)
    torch.testing.assert_close(actual, original, atol=1e-7, rtol=1e-6)
    assert torch.equal(actual.argmax(1), original.argmax(1))


def test_support_permutation_invariance():
    head, z, labels = episode()
    permutation = torch.randperm(15)
    other = VisualEpisodeHead(head_config()).fit(z[:15][permutation], labels[permutation], 5)
    torch.testing.assert_close(head.proker(z[15:], 5., 1., 1.), other.proker(z[15:], 5., 1., 1.),
                               atol=1e-6, rtol=1e-5)


def head_config():
    return load_config(Path(__file__).resolve().parents[1]/"sceg_original/configs/cub.yaml")["model"]


def test_query_independence_and_chunks():
    head, z, _ = episode()
    query = z[15:]
    together = head.proker(query, 5., 1., .5)
    chunks = torch.cat([head.proker(chunk, 5., 1., .5) for chunk in query.split(13)])
    singleton = head.proker(query[7:8], 5., 1., .5)
    torch.testing.assert_close(together, chunks, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(together[7:8], singleton, atol=1e-6, rtol=1e-5)


def test_linear_ridge_primal_dual():
    head, z, _ = episode()
    # Smaller dimensional subspace makes a primal reference cheap without changing the API.
    head.us = head.us[:, :12]
    from unittest.mock import patch
    uq = kernel_features(z[15:])[:, :12]
    with patch("experiments.sceg_proker.head.kernel_features", return_value=uq):
        dual = head.ridge(z[15:], .1)
    primal = torch.linalg.solve(head.us.T@head.us+.1*torch.eye(12), head.us.T@head.targets)
    torch.testing.assert_close(dual, uq@primal, atol=1e-6, rtol=1e-5)


def test_positive_kernel_regularization():
    head, z, _ = episode()
    for value in (0., -1., float("nan")):
        with pytest.raises(ValueError):
            head.proker(z[15:], value, 1., 1.)
        with pytest.raises(ValueError):
            head.proker(z[15:], 1., value, 1.)
