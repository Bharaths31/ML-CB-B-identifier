import torch
import torch.nn as nn

from .config import (BINARY_DIM, BREED_DIM, CBAM_AFTER_STAGE, DROPOUT,
                     NUM_BUFFALO_BREEDS, NUM_CATTLE_BREEDS)
from .efficientnet_lite import EfficientNetLite, load_backbone_weights
from .cbam import build_attention


class BreedClassifier(nn.Module):
    def __init__(self, backbone="lite2", num_cattle=NUM_CATTLE_BREEDS,
                 num_buffalo=NUM_BUFFALO_BREEDS, cbam_stage=CBAM_AFTER_STAGE,
                 attention="cbam", activation="relu6", pretrained_path=None,
                 dropout=DROPOUT):
        super().__init__()
        self.backbone = EfficientNetLite(arch=backbone, activation=activation)
        self.cbam_stage = cbam_stage
        stage_channels = self.backbone.stage_channels[cbam_stage]
        self.attention = build_attention(attention, stage_channels)
        feature_dim = self.backbone.head_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.binary_head = nn.Sequential(
            nn.Linear(feature_dim, BINARY_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(BINARY_DIM, 2),
        )
        self.cattle_head = nn.Sequential(
            nn.Linear(feature_dim, BREED_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(BREED_DIM, num_cattle),
        )
        self.buffalo_head = nn.Sequential(
            nn.Linear(feature_dim, BREED_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(BREED_DIM, num_buffalo),
        )

        if pretrained_path:
            n = load_backbone_weights(self.backbone, pretrained_path)
            print(f"[model] loaded {n} tensors from {pretrained_path}")

    def forward_features(self, x):
        feats = self.backbone.forward_until(x, self.cbam_stage)
        feats = self.attention(feats)
        feats = self.backbone.forward_from(feats, self.cbam_stage + 1)
        return self.avg_pool(feats).flatten(1)

    def forward(self, x):
        pooled = self.forward_features(x)
        return {
            "binary": self.binary_head(pooled),
            "cattle": self.cattle_head(pooled),
            "buffalo": self.buffalo_head(pooled),
            "features": pooled,
        }

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.attention.parameters():
            p.requires_grad = False

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def backbone_eval(self):
        self.backbone.eval()
        self.attention.eval()

    def backbone_train(self):
        self.backbone.train()
        self.attention.train()