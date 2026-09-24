import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (BINARY_DIM, BREED_DIM, CBAM_AFTER_STAGE, COSINE_SCALE,
                     DROPOUT, NUM_BUFFALO_BREEDS, NUM_CATTLE_BREEDS,
                     PROJECTION_DIM)
from .efficientnet_lite import EfficientNetLite, load_backbone_weights
from .cbam import build_attention


class CosineHead(nn.Module):
    """Normalised-feature scaled-cosine classifier (ArcFace-style).

    Forward returns ``scale * cos(theta)`` with NO additive margin, so inference
    and export are plain scaled-cosine logits. The angular margin is applied to
    the target class only during training, inside the loss (see
    ``train._apply_cosine_margin``), which keeps the exported graph margin-free.
    """

    def __init__(self, in_features, out_features, scale=COSINE_SCALE):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)

    def forward(self, x):
        cos = F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))
        return self.scale * cos


def _make_breed_head(feature_dim, num_classes, dropout, cosine=False,
                     scale=COSINE_SCALE):
    layers = [
        nn.Linear(feature_dim, BREED_DIM),
        nn.BatchNorm1d(BREED_DIM),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(BREED_DIM, BREED_DIM // 2),
        nn.BatchNorm1d(BREED_DIM // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
    ]
    layers.append(CosineHead(BREED_DIM // 2, num_classes, scale) if cosine
                  else nn.Linear(BREED_DIM // 2, num_classes))
    return nn.Sequential(*layers)


class BreedClassifier(nn.Module):
    def __init__(self, backbone="lite2", num_cattle=NUM_CATTLE_BREEDS,
                 num_buffalo=NUM_BUFFALO_BREEDS, cbam_stage=CBAM_AFTER_STAGE,
                 attention="cbam", activation="relu6", pretrained_path=None,
                 dropout=DROPOUT, cosine_head=False, cosine_scale=COSINE_SCALE):
        super().__init__()
        self.cosine_head = bool(cosine_head)
        self.cosine_scale = float(cosine_scale)
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
        self.cattle_head = _make_breed_head(feature_dim, num_cattle, dropout,
                                            self.cosine_head, self.cosine_scale)
        self.buffalo_head = _make_breed_head(feature_dim, num_buffalo, dropout,
                                             self.cosine_head, self.cosine_scale)

        # Auxiliary projection head used only by the supervised-contrastive
        # loss (never exported). The 1280-d pooled features were previously
        # returned but unused; this gives the model an explicit objective to
        # cluster embeddings by breed, which is what separates near-identical
        # indigenous breeds.
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, PROJECTION_DIM),
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
            "embedding": self.projection_head(pooled),
        }

    @torch.no_grad()
    def predict(self, x):
        """Soft species routing: never let a hard binary argmax discard the
        other head. Returns the combined 75-class breed distribution, built as
        p(species) * softmax(breed head for that species).
        """
        out = self.forward(x)
        p_species = F.softmax(out["binary"], dim=1)
        p_cattle = F.softmax(out["cattle"], dim=1)
        p_buffalo = F.softmax(out["buffalo"], dim=1)
        return torch.cat([p_species[:, 0:1] * p_cattle,
                          p_species[:, 1:2] * p_buffalo], dim=1)

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