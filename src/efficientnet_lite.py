import torch
import torch.nn as nn

LITE_ARCHS = {
    "lite2": {
        "stem_channels": 32,
        "stages": [
            {"repeat": 1, "kernel": 3, "stride": 1, "expand": 1, "in": 32,  "out": 16},
            {"repeat": 3, "kernel": 3, "stride": 2, "expand": 6, "in": 16,  "out": 24},
            {"repeat": 3, "kernel": 5, "stride": 2, "expand": 6, "in": 24,  "out": 48},
            {"repeat": 4, "kernel": 3, "stride": 2, "expand": 6, "in": 48,  "out": 88},
            {"repeat": 4, "kernel": 5, "stride": 2, "expand": 6, "in": 88,  "out": 120},
            {"repeat": 5, "kernel": 5, "stride": 2, "expand": 6, "in": 120, "out": 208},
            {"repeat": 1, "kernel": 3, "stride": 1, "expand": 6, "in": 208, "out": 352},
        ],
        "head_channels": 1280,
    },
    "lite4": {
        "stem_channels": 32,
        "stages": [
            {"repeat": 1, "kernel": 3, "stride": 1, "expand": 1, "in": 32,  "out": 24},
            {"repeat": 4, "kernel": 3, "stride": 2, "expand": 6, "in": 24,  "out": 32},
            {"repeat": 4, "kernel": 5, "stride": 2, "expand": 6, "in": 32,  "out": 56},
            {"repeat": 6, "kernel": 3, "stride": 2, "expand": 6, "in": 56,  "out": 112},
            {"repeat": 6, "kernel": 5, "stride": 2, "expand": 6, "in": 112, "out": 160},
            {"repeat": 8, "kernel": 5, "stride": 2, "expand": 6, "in": 160, "out": 272},
            {"repeat": 1, "kernel": 3, "stride": 1, "expand": 6, "in": 272, "out": 448},
        ],
        "head_channels": 1280,
    },
}


def make_activation(name="relu6"):
    name = (name or "relu6").lower()
    if name in ("relu6", "relu6_inplace"):
        return nn.ReLU6(inplace=True)
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unknown activation {name}")


class MBConvBlock(nn.Module):
    def __init__(self, cin, cout, kernel, stride, expand, activation):
        super().__init__()
        self.activation = activation
        self._add = stride == 1 and cin == cout
        expanded = cin * expand
        self._depthwise_conv = nn.Conv2d(
            expanded, expanded, kernel, stride=stride, padding=kernel // 2,
            groups=expanded, bias=False)
        self._bn1 = nn.BatchNorm2d(expanded)
        if expand != 1:
            self._expand_conv = nn.Conv2d(cin, expanded, 1, bias=False)
            self._bn0 = nn.BatchNorm2d(expanded)
        self._project_conv = nn.Conv2d(expanded, cout, 1, bias=False)
        self._bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        if self._add:
            residual = x
        if hasattr(self, "_expand_conv"):
            x = self.activation(self._bn0(self._expand_conv(x)))
        x = self.activation(self._bn1(self._depthwise_conv(x)))
        x = self._bn2(self._project_conv(x))
        if self._add:
            x = x + residual
        return x


class EfficientNetLite(nn.Module):
    def __init__(self, arch="lite2", activation="relu6"):
        super().__init__()
        spec = LITE_ARCHS[arch]
        self.arch = arch
        self.head_channels = spec["head_channels"]
        self.stage_channels = [s["out"] for s in spec["stages"]]
        act = make_activation(activation)

        self.stem = nn.Sequential(
            nn.Conv2d(3, spec["stem_channels"], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(spec["stem_channels"]),
            act,
        )
        self.blocks = nn.ModuleList()
        for stage in spec["stages"]:
            stage_modules = nn.ModuleList()
            for rep in range(stage["repeat"]):
                stage_modules.append(MBConvBlock(
                    cin=stage["in"] if rep == 0 else stage["out"],
                    cout=stage["out"],
                    kernel=stage["kernel"],
                    stride=stage["stride"] if rep == 0 else 1,
                    expand=stage["expand"],
                    activation=act,
                ))
            self.blocks.append(stage_modules)
        self.head = nn.Sequential(
            nn.Conv2d(spec["stages"][-1]["out"], spec["head_channels"], 1, bias=False),
            nn.BatchNorm2d(spec["head_channels"]),
            act,
        )

    def forward_until(self, x, stage_index):
        x = self.stem(x)
        for i in range(stage_index + 1):
            for block in self.blocks[i]:
                x = block(x)
        return x

    def forward_from(self, x, stage_index):
        for i in range(stage_index, len(self.blocks)):
            for block in self.blocks[i]:
                x = block(x)
        x = self.head(x)
        return x

    def forward(self, x):
        last = len(self.blocks) - 1
        return self.forward_from(self.forward_until(x, last), last + 1)


def load_backbone_weights(backbone, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint contents in {path}: {type(checkpoint)}")
    expected = set(backbone.state_dict().keys())
    provided = set(checkpoint.keys())
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    allowed_missing = set()
    allowed_unexpected = {"fc.weight", "fc.bias"}
    hard_missing = [k for k in missing if k not in allowed_missing]
    hard_unexpected = [k for k in unexpected if k not in allowed_unexpected]
    if hard_missing or hard_unexpected:
        raise ValueError(
            f"Weight mismatch for {path}:\n"
            f"  missing: {hard_missing[:8]}\n"
            f"  unexpected: {hard_unexpected[:8]}")
    backbone.load_state_dict(checkpoint, strict=False)
    return len(provided)