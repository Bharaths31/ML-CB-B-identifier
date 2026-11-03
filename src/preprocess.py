"""Inference-time preprocessing shared by the GUI and the Flask API.

Kept separate from ``src/data_pipeline.py`` on purpose: that module imports
pandas/tqdm and the whole dataset stack, which neither the API server nor the
model tester needs just to resize one image.

The eval chain used for training centre-crops a 4:3 phone photo and therefore
loses ~25% of its width -- exactly the region that carries horns, the udder and
the tail. ``PadToSquare`` keeps the entire body: resize the LONG side to
``size`` and pad the short side with the ImageNet mean (which becomes ~0 after
``Normalize``), so the network sees the whole animal against "no signal".

``src/data_pipeline.py`` re-exports ``PadToSquare`` as ``_PadToSquare`` so the
training/eval transforms and every inference entry point share one
implementation.
"""

from PIL import Image
from torchvision import transforms

from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


class PadToSquare:
    """Resize the LONG side to ``size`` and pad the short side to a square."""

    def __init__(self, size=IMAGE_SIZE, fill=None):
        self.size = int(size)
        self.fill = fill or tuple(int(round(m * 255)) for m in IMAGENET_MEAN)

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas

    def __repr__(self):
        return f"PadToSquare({self.size}, fill={self.fill})"


def inference_transform(pad=None, size=IMAGE_SIZE):
    """Build the inference chain.

    ``pad=True``  -> PadToSquare(size) + ToTensor + Normalize  (whole frame)
    ``pad=False`` -> Resize(shortest side) + CenterCrop         (legacy parity)

    ``pad=None`` follows ``config.INFERENCE_PAD_TO_SQUARE``.
    """
    if pad is None:
        from .config import INFERENCE_PAD_TO_SQUARE
        pad = INFERENCE_PAD_TO_SQUARE

    if pad:
        blocks = [PadToSquare(size)]
    else:
        blocks = [transforms.Resize(size), transforms.CenterCrop(size)]
    blocks += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(blocks)


def describe(transform):
    """Human-readable chain for startup/debug logs."""
    ops = getattr(transform, "transforms", [transform])
    return " -> ".join(repr(op) for op in ops)
