"""Inductive visual ProKeR migration; no changes to the SCEG model."""
import math
import torch
from torch.nn import functional as F
from models.modules.cosine_classifier import CosineClassifier
from utils.config import ModelConfig


def kernel_features(features):
    """features [N,3,640], unnormalized post-FC stage2/3/4 embeddings."""
    return F.normalize(features.float(), dim=-1, eps=1e-12).flatten(1)/math.sqrt(3)


def rbf_kernel(a, b, beta):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    return (-beta*(1-a.float()@b.float().T)).exp()


def residual_coefficients(k_ss, residual, ridge_lambda):
    """Author code: (I+K/lambda) A = residual, NOT (K+lambda I) A = residual."""
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be finite and positive")
    identity = torch.eye(len(k_ss), device=k_ss.device, dtype=k_ss.dtype)
    return torch.linalg.solve(identity+k_ss/ridge_lambda, residual)


class VisualEpisodeHead:
    """Fit on support alone; queries are independent and labels are never accepted."""
    def __init__(self, model_config):
        self.classifier = CosineClassifier(ModelConfig.from_dict(model_config))

    def fit(self, support, labels, way):
        if support.ndim != 3 or support.shape[1:] != (3, 640):
            raise ValueError("Expected three post-FC 640-dimensional branches")
        if labels.shape != (len(support),) or labels.dtype != torch.long:
            raise ValueError("One integer episode label is required per support")
        if set(labels.tolist()) != set(range(way)):
            raise ValueError("Support must cover exactly the episode classes")
        self.support, self.labels, self.way = support.float(), labels, way
        # Mean unnormalized samples FIRST, then use the ORIGINAL episode normalizer.
        prototypes = torch.stack([self.support[labels == c].mean(0) for c in range(way)])
        self.prototypes = [self.classifier.normalize_episode(prototypes[:, s]) for s in range(3)]
        self.us = kernel_features(self.support)
        self.targets = F.one_hot(labels, num_classes=way).float()
        return self

    def baseline(self, query):
        scores = [torch.einsum("qd,wd->qw", self.classifier.normalize_episode(query[:, s]),
                               self.prototypes[s]) for s in range(3)]
        # Preserve original stage3 + stage4 + stage2 summation order.
        return (scores[1]+scores[2]+scores[0])/3

    def proker(self, query, beta, ridge_lambda, residual_weight):
        if not math.isfinite(residual_weight) or residual_weight < 0:
            raise ValueError("residual_weight must be finite and nonnegative")
        f0q = self.baseline(query)
        if residual_weight == 0:
            return f0q
        kss = rbf_kernel(self.us, self.us, beta)
        a = residual_coefficients(kss, self.targets-self.baseline(self.support), ridge_lambda)
        return f0q+residual_weight*rbf_kernel(kernel_features(query), self.us, beta)@a

    def ridge(self, query, ridge_lambda):
        """Linear ridge on the SAME normalized concatenation; no bias or residual prior."""
        if not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be finite and positive")
        gram = self.us@self.us.T
        a = torch.linalg.solve(gram+ridge_lambda*torch.eye(len(gram), device=gram.device),
                               self.targets)
        return kernel_features(query)@self.us.T@a
