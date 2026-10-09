"""SCEG ResNet12, official models/backbones/ResNet.py:18–202.

Paper §4.1.2: four residual stages, each with three convolutions.
Input [B,3,84,84]; output stage2/3/4 [B,160,21,21], [B,320,10,10],
[B,640,5,5]. Block: pool(LeakyReLU(conv_stack(x)+projection(x))).
DropBlock sampling and eval-time counter increments preserve official behavior.
"""
import torch
from torch import nn
from torch.nn import functional as functional


class DropBlock(nn.Module):
    """Contiguous dropout. y = mask*x*(num_elements/sum(mask))."""
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, feature_map, gamma):
        if not self.training:
            return feature_map
        batch, channels, height, width = feature_map.shape
        assert height >= self.block_size and width >= self.block_size
        # Official draws from CPU RNG, then transfers to CUDA. Preserve that order.
        mask = torch.distributions.Bernoulli(gamma).sample(
            (batch, channels, height-self.block_size+1, width-self.block_size+1)
        ).to(device=feature_map.device)
        block_mask = self._compute_block_mask(mask)
        # No new epsilon: all-zero masks retain the author's NaN behavior.
        return block_mask * feature_map * (block_mask.numel()/block_mask.sum())

    def _compute_block_mask(self, mask):
        left_padding = (self.block_size-1)//2
        right_padding = self.block_size//2
        active_indices = mask.nonzero()
        block_count = active_indices.shape[0]
        offsets_spatial = torch.stack([
            torch.arange(self.block_size).view(-1, 1).expand(
                self.block_size, self.block_size).reshape(-1),
            torch.arange(self.block_size).repeat(self.block_size),
        ]).t().to(mask.device)
        offsets = torch.cat([
            torch.zeros(self.block_size**2, 2, device=mask.device).long(),
            offsets_spatial.long(),
        ], dim=1)
        padded_mask = functional.pad(
            mask, (left_padding, right_padding, left_padding, right_padding))
        if block_count > 0:
            block_indices = active_indices.repeat(self.block_size**2, 1) + offsets.repeat(
                block_count, 1).view(-1, 4)
            padded_mask[block_indices[:, 0], block_indices[:, 1],
                        block_indices[:, 2], block_indices[:, 3]] = 1
        return 1-padded_mask


class ResidualBlock(nn.Module):
    """[B,Cin,H,W] → [B,Cout,floor(H/2),floor(W/2)]."""
    def __init__(self, input_channels, output_channels, config, drop_block):
        super().__init__()
        kernel = config.convolution_kernel
        padding = kernel//2
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels, eps=config.batchnorm_eps,
                                  momentum=config.batchnorm_momentum)
        self.relu = nn.LeakyReLU(config.negative_slope)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels, eps=config.batchnorm_eps,
                                  momentum=config.batchnorm_momentum)
        self.conv3 = nn.Conv2d(output_channels, output_channels, kernel, padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels, eps=config.batchnorm_eps,
                                  momentum=config.batchnorm_momentum)
        self.maxpool = nn.MaxPool2d(config.pool_stride)
        self.downsample = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels, eps=config.batchnorm_eps,
                            momentum=config.batchnorm_momentum),
        )
        self.drop_rate = config.dropout
        self.num_batches_tracked = 0  # Official plain counter, including eval forwards.
        self.drop_block = drop_block
        self.block_size = config.dropblock_size
        self.DropBlock = DropBlock(self.block_size)
        self.ramp_batches = config.dropblock_ramp_batches

    def forward(self, feature_map):
        self.num_batches_tracked += 1
        residual_feature = self.downsample(feature_map)
        feature_conv1 = self.relu(self.bn1(self.conv1(feature_map)))
        feature_conv2 = self.relu(self.bn2(self.conv2(feature_conv1)))
        feature_conv3 = self.bn3(self.conv3(feature_conv2))
        pooled_feature = self.maxpool(self.relu(feature_conv3+residual_feature))
        if self.drop_rate > 0:
            if self.drop_block:
                spatial_size = pooled_feature.shape[2]
                keep_rate = max(1-self.drop_rate/self.ramp_batches*self.num_batches_tracked,
                                1-self.drop_rate)
                gamma = ((1-keep_rate)/self.block_size**2 * spatial_size**2
                         / (spatial_size-self.block_size+1)**2)
                pooled_feature = self.DropBlock(pooled_feature, gamma)
            else:
                pooled_feature = functional.dropout(pooled_feature, p=self.drop_rate,
                                                      training=self.training, inplace=True)
        return pooled_feature


class ResNet12(nn.Module):
    def __init__(self, config):
        super().__init__()
        previous_channels = config.input_channels
        for stage_index, output_channels in enumerate(config.stage_channels, start=1):
            setattr(self, f"layer{stage_index}", nn.Sequential(ResidualBlock(
                previous_channels, output_channels, config, drop_block=stage_index >= 3)))
            previous_channels = output_channels
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, images):
        assert images.ndim == 4 and images.shape[-1] == images.shape[-2]
        feature_stage1 = self.layer1(images)  # [B,64,42,42]
        feature_stage2 = self.layer2(feature_stage1)  # [B,160,21,21]
        feature_stage3 = self.layer3(feature_stage2)  # [B,320,10,10]
        feature_stage4 = self.layer4(feature_stage3)  # [B,640,5,5]
        return feature_stage2, feature_stage3, feature_stage4
