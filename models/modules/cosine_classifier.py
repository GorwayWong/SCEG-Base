"""Cosine classifiers: paper Eq15–17, official SCEG.py:83–96.

Input support-first embeddings [W*(S+Q),D]; output [W*Q,W].
p_c=mean_s(z_cs); score_qc=<n(z_q),n(p_c)>.
Episode n(z)=z/sqrt(sum(z²)+eps). Global n uses F.normalize's
max(norm,eps). These two semantics must remain distinct (D13).
"""
import torch
from torch import nn
from torch.nn import functional as functional


class CosineClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.episode_eps = config.episode_norm_eps
        self.global_eps = config.global_norm_eps

    def normalize_episode(self, embedding):
        assert embedding.ndim == 2
        return embedding/(embedding.pow(2).sum(dim=1)+self.episode_eps).sqrt().unsqueeze(1)

    def forward(self, embedding, way, shot):
        assert way > 0 and shot > 0 and embedding.ndim == 2
        support_count = way*shot
        assert embedding.shape[0] > support_count
        assert (embedding.shape[0]-support_count) % way == 0
        support_prototype = embedding[:support_count].reshape(way, shot, -1).mean(dim=1)
        query_feature = embedding[support_count:]
        assert support_prototype.shape[-1] == query_feature.shape[-1]
        return torch.einsum("qd,wd->qw", self.normalize_episode(query_feature),
                            self.normalize_episode(support_prototype))

    def global_scores(self, embedding, class_representatives):
        assert embedding.ndim == class_representatives.ndim == 2
        assert embedding.shape[-1] == class_representatives.shape[-1]
        return torch.einsum("bd,kd->bk",
            functional.normalize(embedding, p=2, dim=-1, eps=self.global_eps),
            functional.normalize(class_representatives, p=2, dim=-1, eps=self.global_eps))
