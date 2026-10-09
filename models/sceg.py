"""Compose SCEG only; official models/SCEG.py:44–113.

Paper §3: backbone→SC→feature transform→episode/global cosine.
Input [W*(S+Q),3,84,84], support first. Output three [W*Q,W]
cosines (official order stage3,stage4,stage2) and [B,K] global cosine.
Test score=sum of three cosines (range [-3,3]), prediction=argmax.
"""
import math
from typing import NamedTuple
import torch
from torch import nn
from .backbone.resnet12 import ResNet12
from .modules.sc import SC
from .modules.transform_fc import TransformFC
from .modules.cosine_classifier import CosineClassifier


class SCEGOutput(NamedTuple):
    stage3_cosine: torch.Tensor
    stage4_cosine: torch.Tensor
    stage2_cosine: torch.Tensor
    global_cosine: torch.Tensor

    @property
    def scores(self):
        return self.stage3_cosine+self.stage4_cosine+self.stage2_cosine


class SCEG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Official registration names/order retained for direct state_dict compatibility.
        self.feature_extractor = ResNet12(config)
        self.fc_t = TransformFC(config)
        self.fc_s = TransformFC(config)
        self.fc_l = TransformFC(config)
        self.class_representative = nn.Linear(config.feature_dim, config.base_classes, bias=False)
        self.SC = SC(config)
        self.classifier = CosineClassifier(config)
        # Preserve SCEG's override of backbone initialization and unchanged Conv bias.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                scale = module.kernel_size[0]*module.kernel_size[1]*module.out_channels
                module.weight.data.normal_(0, math.sqrt(2/scale))
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def encode(self, images):
        assert images.ndim == 4 and images.shape[1] == self.config.input_channels
        assert images.shape[-2:] == (self.config.image_size, self.config.image_size)
        feature_stage2, feature_stage3, feature_stage4 = self.feature_extractor(images)
        embedding_stage2, embedding_stage3, embedding_stage4 = self.SC(
            feature_stage2, feature_stage3, feature_stage4)
        return (self.fc_t(embedding_stage2), self.fc_s(embedding_stage3),
                self.fc_l(embedding_stage4))

    def classify(self, embeddings, way, shot):
        embedding_stage2, embedding_stage3, embedding_stage4 = embeddings
        cosine_stage3 = self.classifier(embedding_stage3, way, shot)
        cosine_stage2 = self.classifier(embedding_stage2, way, shot)
        cosine_stage4 = self.classifier(embedding_stage4, way, shot)
        global_cosine = self.classifier.global_scores(
            embedding_stage4, self.class_representative.weight)
        return SCEGOutput(cosine_stage3, cosine_stage4, cosine_stage2, global_cosine)

    def forward(self, images, way=None, shot=None):
        return self.classify(self.encode(images),
            self.config.train_way if way is None else way,
            self.config.train_shot if shot is None else shot)

    def meta_test(self, images, way, shot):
        return self(images, way=way, shot=shot).scores.argmax(dim=1)

    def expected_inactive_parameters(self):
        names = {name for name, _ in self.named_parameters()
                 if name.startswith(("SC.t_s.mlp.", "SC.t_t.mlp.", "SC.t_l.mlp."))}
        if self.config.dataset not in self.config.stage3_to_stage2_datasets:
            names.update(name for name, _ in self.named_parameters()
                         if name.startswith("SC.g_st."))
        return names
