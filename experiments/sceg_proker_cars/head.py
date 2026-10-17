"""Minimal visual head from Dogs commit 6facdc199acda7b17a2b3b6cda5cc353a39878c0.

Differences: fixed class-major support order permits reshape mean; validation is
at the input boundary; normalized query/support and prototypes are shared once.
The reference kernel, residual solve, normalization and score scale are unchanged.
"""
import math
import torch
from torch.nn import functional as F


def kernel_features(features):
    return F.normalize(features.float(), dim=-1, eps=1e-12).flatten(1) / math.sqrt(3)


def prepare(features, way, shot, classifier):
    support = features[:way * shot]
    query = features[way * shot:]
    # Average raw support BEFORE the original episode normalization.
    prototypes = support.reshape(way, shot, 3, 640).mean(1)
    prototypes = [classifier.normalize_episode(prototypes[:, s]) for s in range(3)]
    labels = torch.arange(way, device=features.device).repeat_interleave(shot)
    return support, query, prototypes, kernel_features(support), kernel_features(query), F.one_hot(labels, way).float()


def original_scores(features, prototypes, classifier):
    scores = [torch.einsum('qd,wd->qw', classifier.normalize_episode(features[:, s]),
                           prototypes[s]) for s in range(3)]
    return scores[1] + scores[2] + scores[0]


def linear_ridge(prepared, ridge_lambda):
    _, _, _, us, uq, targets = prepared
    gram = us @ us.T
    a = torch.linalg.solve(gram + ridge_lambda * torch.eye(len(us), device=us.device), targets)
    return uq @ us.T @ a


def proker(prepared, classifier, beta, ridge_lambda, residual_weight):
    support, query, prototypes, us, uq, targets = prepared
    f0q = original_scores(query, prototypes, classifier) / 3
    if residual_weight == 0:
        return f0q
    f0s = original_scores(support, prototypes, classifier) / 3
    kss = (-beta * (1 - us @ us.T)).exp()
    kqs = (-beta * (1 - uq @ us.T)).exp()
    # Reference convention; do not replace with solve(K + lambda I, residual).
    a = torch.linalg.solve(torch.eye(len(us), device=us.device) + kss / ridge_lambda, targets - f0s)
    return f0q + residual_weight * kqs @ a
