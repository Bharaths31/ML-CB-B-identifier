import io
import json
from PIL import Image
from PIL.ExifTags import TAGS as EXIF_TAGS

def extract_image_metadata(image_bytes):
    """Extract detailed metadata from raw image bytes."""
    meta = {}
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Basic dimensions
        meta["width"] = img.width
        meta["height"] = img.height
        meta["aspect_ratio"] = f"{img.width}:{img.height}"
        from math import gcd
        g = gcd(img.width, img.height)
        meta["aspect_ratio_simplified"] = f"{img.width // g}:{img.height // g}"
        meta["megapixels"] = round((img.width * img.height) / 1_000_000, 2)
        meta["orientation"] = "Landscape" if img.width > img.height else (
            "Portrait" if img.height > img.width else "Square")

        # Format & color
        meta["format"] = img.format or "Unknown"
        meta["mode"] = img.mode
        mode_map = {"L": "Grayscale", "LA": "Grayscale+Alpha", "RGB": "RGB",
                    "RGBA": "RGB+Alpha", "CMYK": "CMYK", "P": "Palette",
                    "1": "Binary", "I": "32-bit Integer", "F": "32-bit Float"}
        meta["color_space"] = mode_map.get(img.mode, img.mode)
        meta["channels"] = len(img.getbands())
        meta["bands"] = list(img.getbands())

        if img.mode in ("I", "I;16"):
            meta["bit_depth"] = 16
        elif img.mode == "F":
            meta["bit_depth"] = 32
        elif img.mode == "1":
            meta["bit_depth"] = 1
        else:
            meta["bit_depth"] = 8

        # File size
        meta["file_size_bytes"] = len(image_bytes)
        meta["file_size_kb"] = round(len(image_bytes) / 1024, 2)
        meta["file_size_mb"] = round(len(image_bytes) / (1024 * 1024), 3)

        # DPI
        dpi = img.info.get("dpi")
        if dpi:
            meta["dpi_x"] = round(dpi[0], 1)
            meta["dpi_y"] = round(dpi[1], 1)

        # EXIF data
        exif_data = {}
        try:
            raw_exif = img._getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = EXIF_TAGS.get(tag_id, str(tag_id))
                    if isinstance(value, bytes) and len(value) > 100:
                        continue
                    try:
                        json.dumps(value)
                        exif_data[tag_name] = value
                    except (TypeError, ValueError):
                        exif_data[tag_name] = str(value)
        except Exception:
            pass
        meta["exif"] = exif_data

        if img.width * img.height < 500_000:
            try:
                colors = img.convert("RGB").getcolors(maxcolors=100_000)
                meta["unique_colors"] = len(colors) if colors else "100000+"
            except Exception:
                meta["unique_colors"] = "N/A"
        else:
            meta["unique_colors"] = "N/A (image too large to count)"

    except Exception as e:
        meta["error"] = str(e)

    return meta
