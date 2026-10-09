"""Feature transform: paper §3.3; official SCEG.py:27–41.

Input/output [B,D]. z'=alpha*z+(1-alpha)*ELU(Linear(BN(z))).
Official alpha=0.3, exposed in config for auditing, not changed in baseline.
"""
from torch import nn


class TransformFC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha = config.transform_alpha
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(config.feature_dim, eps=config.batchnorm_eps,
                            momentum=config.batchnorm_momentum),
            nn.Linear(config.feature_dim, config.feature_dim), nn.ELU(inplace=True),
        )

    def forward(self, embedding):
        assert embedding.ndim == 2
        return self.alpha*embedding + (1-self.alpha)*self.mlp(embedding)
