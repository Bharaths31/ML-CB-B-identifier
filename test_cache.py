import io
import os
from PIL import Image

def test_cache_logic(path):
    with open(path, "rb") as f:
        b = f.read()
    img = Image.open(io.BytesIO(b)).convert("RGB")
    print(img.size)

if __name__ == '__main__':
    # Just a test snippet
    print("Done")
