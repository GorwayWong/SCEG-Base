"""SC from paper §3.3, official models/SC.py:24–126.

Input: stage2/3/4 [B,160,21,21], [B,320,10,10], [B,640,5,5].
Output: stage2/3/4 embeddings, each [B,640].
F3' = F3*sigmoid(g43(resize(F4))) + self3(F3).
F2' = F2*sigmoid(g42(resize(F4))) + self2(F2), plus the stage3
gate only for CUB/Dogs. F4'=self4(F4). Then conv/BN/ReLU/max-pool.
Unused projection MLPs and discarded Cars/Flowers gate are retained (D01/D02).
"""
from torch import nn
from torch.nn import functional as functional


class PointwiseProjection(nn.Module):
    def __init__(self, input_channels, config):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, config.feature_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(config.feature_dim, eps=config.batchnorm_eps,
                                  momentum=config.projection_bn_momentum)
        self.relu = nn.ReLU()

    def forward(self, feature_map):
        return self.relu(self.bn(self.conv(feature_map)))


class SelfEnhancement(nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        for layer_index in (1, 2):
            setattr(self, f"layer{layer_index}", nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.BatchNorm2d(channels, eps=config.batchnorm_eps,
                                momentum=config.batchnorm_momentum),
                nn.ELU(inplace=True),
            ))

    def forward(self, feature_map):
        return self.layer2(self.layer1(feature_map))


class SpatialTransform(nn.Module):
    """[B,C,H,W] → [B,D]; max_pool(ReLU(BN(conv1x1(F))))."""
    def __init__(self, input_channels, config):
        super().__init__()
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.conv_block = nn.Sequential(PointwiseProjection(input_channels, config))
        # Registered but never called by the author. Preserve state_dict and gradients.
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(config.feature_dim, eps=config.batchnorm_eps,
                            momentum=config.batchnorm_momentum),
            nn.Linear(config.feature_dim, config.feature_dim), nn.ELU(inplace=True),
        )

    def forward(self, feature_map):
        projected_feature = self.conv_block(feature_map)  # [B,C,H,W] → [B,D,H,W]
        pooled_feature = self.max(projected_feature)  # [B,D,H,W] → [B,D,1,1]
        return pooled_feature.flatten(start_dim=1)  # [B,D]


class SC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        _, stage2_channels, stage3_channels, stage4_channels = config.stage_channels
        self.self_t = SelfEnhancement(stage2_channels, config)
        self.self_s = SelfEnhancement(stage3_channels, config)
        self.self_l = SelfEnhancement(stage4_channels, config)
        self.g_ls = self._gate(stage4_channels, config)
        self.g_lt = self._gate(stage4_channels, config)
        self.g_st = self._gate(stage3_channels, config)
        self.t_s = SpatialTransform(stage3_channels, config)
        self.t_t = SpatialTransform(stage2_channels, config)
        self.t_l = SpatialTransform(stage4_channels, config)

    @staticmethod
    def _gate(input_channels, config):
        hidden = config.gate_hidden_channels
        return nn.Sequential(
            nn.Conv2d(input_channels, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden, eps=config.batchnorm_eps,
                            momentum=config.batchnorm_momentum),
            nn.ReLU(inplace=True), nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, feature_stage2, feature_stage3, feature_stage4):
        for feature_map, channels in zip(
                (feature_stage2, feature_stage3, feature_stage4), self.config.stage_channels[1:]):
            assert feature_map.ndim == 4 and feature_map.shape[1] == channels
            assert feature_map.shape[-2] == feature_map.shape[-1]
        upsampled_4_to_3 = functional.interpolate(feature_stage4,
            size=(feature_stage3.shape[-1],)*2, mode="bilinear", align_corners=False)
        gate_4_to_3 = self.g_ls(upsampled_4_to_3).sigmoid()  # [B,1,10,10]
        enhanced_stage3 = feature_stage3*gate_4_to_3 + self.self_s(feature_stage3)
        upsampled_4_to_2 = functional.interpolate(feature_stage4,
            size=(feature_stage2.shape[-1],)*2, mode="bilinear", align_corners=False)
        upsampled_3_to_2 = functional.interpolate(feature_stage3,
            size=(feature_stage2.shape[-1],)*2, mode="bilinear", align_corners=False)
        gate_4_to_2 = self.g_lt(upsampled_4_to_2).sigmoid()  # [B,1,21,21]
        gate_3_to_2 = self.g_st(upsampled_3_to_2).sigmoid()  # Computed on all datasets.
        if self.config.dataset in self.config.stage3_to_stage2_datasets:
            enhanced_stage2 = (feature_stage2*gate_4_to_2 + feature_stage2*gate_3_to_2
                               + self.self_t(feature_stage2))
        else:
            enhanced_stage2 = feature_stage2*gate_4_to_2 + self.self_t(feature_stage2)
        enhanced_stage4 = self.self_l(feature_stage4)
        embedding_stage3 = self.t_s(enhanced_stage3)
        embedding_stage2 = self.t_t(enhanced_stage2)
        embedding_stage4 = self.t_l(enhanced_stage4)
        return embedding_stage2, embedding_stage3, embedding_stage4
