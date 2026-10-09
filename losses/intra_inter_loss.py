"""Intra-Inter loss: paper §3.4 Eq20/22; official loss.py:7–41.

Input cosine [B,K], labels [B]; output scalar mean.
L=mean(log(1+sum_pos exp(-lambda*(cos-intra)))
       +log(1+sum_neg exp(lambda*(cos-(intra-inter))))).
Preserves direct exp/log, lambda Parameter (excluded from model optimizer),
and sklearn's binary label_binarize broadcast edge case. Only device/label
construction is adapted to torch; no softplus/logsumexp substitution.
"""
import torch
from torch import nn


class IntraInterLoss(nn.Module):
    def __init__(self, n_way, lamda, inter, intra):
        super().__init__()
        self.n_way = n_way
        self.lamda = nn.Parameter(torch.tensor([lamda], dtype=torch.float32))
        self.inter = inter
        self.intra = intra

    def forward(self, cosine, labels):
        assert cosine.ndim == 2 and cosine.shape[1] == self.n_way
        assert labels.ndim == 1 and labels.shape[0] == cosine.shape[0]
        assert labels.dtype == torch.long
        assert labels.min() >= 0 and labels.max() < self.n_way
        if self.n_way == 2:
            # sklearn label_binarize(classes=[0,1]) returns [B,1], then broadcasts.
            positive_mask = (labels == 1).unsqueeze(1)
        else:
            positive_mask = labels.unsqueeze(1) == torch.arange(self.n_way, device=cosine.device)
        positive_exp = torch.exp(-self.lamda*(cosine-self.intra))
        negative_exp = torch.exp(self.lamda*(cosine-(self.intra-self.inter)))
        positive_sum = torch.where(positive_mask, positive_exp, torch.zeros_like(positive_exp)).sum(1)
        negative_sum = torch.where(~positive_mask, negative_exp, torch.zeros_like(negative_exp)).sum(1)
        return (torch.log(1+positive_sum)+torch.log(1+negative_sum)).mean()
