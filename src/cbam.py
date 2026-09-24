import torch
import torch.nn as nn

from .config import CBAM_IDENTITY_INIT


def _zero_init_last(seq):
    """Zero the last Conv2d weight in a Sequential so its output is 0."""
    for layer in reversed(list(seq)):
        if isinstance(layer, nn.Conv2d):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16, identity_init=CBAM_IDENTITY_INIT):
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.identity_init = identity_init
        if identity_init:
            _zero_init_last(self.mlp)

    def forward(self, x):
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        att = self.mlp(avg) + self.mlp(mx)
        if self.identity_init:
            # gate is 0 at init (zero-init weights), so x*(1+gate) == x.
            gate = 2.0 * self.sigmoid(att) - 1.0
            return x * (1.0 + gate)
        return x * self.sigmoid(att)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, identity_init=CBAM_IDENTITY_INIT):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.identity_init = identity_init
        if identity_init:
            nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.amax(x, dim=1, keepdim=True)
        fused = torch.cat([avg, mx], dim=1)
        att = self.conv(fused)
        if self.identity_init:
            gate = 2.0 * self.sigmoid(att) - 1.0
            return x * (1.0 + gate)
        return x * self.sigmoid(att)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7,
                 identity_init=CBAM_IDENTITY_INIT):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction,
                                                  identity_init=identity_init)
        self.spatial_attention = SpatialAttention(kernel_size,
                                                  identity_init=identity_init)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16, identity_init=CBAM_IDENTITY_INIT):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.identity_init = identity_init
        if identity_init:
            _zero_init_last(self.fc)

    def forward(self, x):
        pooled = self.pool(x)
        att = self.fc[:-1](pooled) if self.identity_init else self.fc(pooled)
        if self.identity_init:
            gate = 2.0 * torch.sigmoid(att) - 1.0
            return x * (1.0 + gate)
        return x * self.fc(pooled)


def build_attention(kind, channels):
    kind = (kind or "cbam").lower()
    if kind == "cbam":
        return CBAM(channels)
    if kind == "se":
        return SEBlock(channels)
    raise ValueError(f"Unknown attention kind {kind}")
