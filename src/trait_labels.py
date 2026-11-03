"""Human-readable labels for the seven morphological trait fields.

Kept free of torch/numpy imports: ``src/breed_knowledge.py`` uses this to build
farmer-facing physical markers, and the advisory endpoint must stay fast.

The vocabulary is the same sorted set produced by
``src.model.build_trait_vocab`` from ``data/breed_traits.json``.
"""

TRAIT_DISPLAY = {
    "hump": {
        "absent": "Absent / Flat",
        "small": "Small",
        "medium": "Medium",
        "large": "Large / Prominent",
        "well_developed": "Well-Developed (Cervicothoracic)",
        "large_well_developed": "Large (Cervicothoracic)",
    },
    "horn": {
        "absent": "Absent / Polled",
        "short": "Short",
        "short_upward": "Short, Upward",
        "short_curved": "Short, Curved",
        "short_curled": "Short, Curled",
        "short_thick": "Short & Thick",
        "short_stumpy": "Short & Stumpy",
        "short_backward": "Short, Backward",
        "stumpy": "Stumpy",
        "medium_curved": "Medium, Curved",
        "medium_curved_upward": "Medium, Curved Upward",
        "medium_upward_curved": "Medium, Upward Curved",
        "long_curved": "Long, Curved",
        "long_flat": "Long & Flat",
        "long_backward_curved": "Long, Backward Curved",
        "curved_backward": "Curved Backward",
        "forward_curved": "Forward Curved",
        "flat_curved": "Flat & Curved",
        "upward_curved": "Upward Curved",
        "heavy_drooping": "Heavy & Drooping",
        "lyre_shaped": "Lyre-Shaped",
        "semicircular": "Semicircular (Half-moon)",
        "sickle_shaped": "Sickle-Shaped",
        "tightly_curled": "Tightly Curled",
    },
    "coat": {
        "black": "Black",
        "jet_black": "Jet Black",
        "white": "White",
        "grey": "Grey",
        "light_grey": "Light Grey",
        "silver_grey": "Silver Grey",
        "white_grey": "White-Grey",
        "greyish_white": "Greyish White",
        "greyish_black": "Greyish Black",
        "black_grey": "Black-Grey",
        "black_brown_grey": "Black / Brown / Grey",
        "black_and_white": "Black & White",
        "brown_white_patches": "Brown with White Patches",
        "fawn": "Fawn",
        "fawn_and_white": "Fawn & White",
        "red_and_white": "Red & White",
        "red_brown": "Red-Brown",
        "red_mottled": "Red Mottled",
        "solid_brown": "Solid Brown",
        "spotted": "Spotted",
    },
    "ear": {
        "drooping": "Drooping",
        "long_pendulous": "Long & Pendulous",
        "medium_horizontal": "Medium, Horizontal",
        "medium_pendulous": "Medium, Pendulous",
        "short_drooping": "Short, Drooping",
        "small_erect": "Small & Erect",
        "small_horizontal": "Small, Horizontal",
        "small_pointed": "Small & Pointed",
    },
    "dewlap": {
        "minimal": "Minimal",
        "small": "Small",
        "moderate": "Moderate",
        "moderate_pendulous": "Moderate & Pendulous",
        "large_pendulous": "Large & Pendulous",
    },
    "face": {
        "broad_concave": "Broad & Concave",
        "broad_convex": "Broad & Convex",
        "broad_flat": "Broad & Flat",
        "long_convex": "Long & Convex",
        "long_narrow": "Long & Narrow",
        "narrow_straight": "Narrow & Straight",
        "short_concave": "Short & Concave",
        "short_straight": "Short & Straight",
        "straight_profile": "Straight Profile",
    },
    "size": {
        "small": "Small / Compact",
        "small_medium": "Small-Medium",
        "medium": "Medium",
        "medium_large": "Medium-Large",
        "large": "Large",
    },
}

# Ordered fallback labels when the trained head vocabulary cannot be aligned
# with data/breed_traits.json (see src.attributes_zeroshot).
TRAIT_GENERIC_LABELS = {
    "hump": ["Absent / Flat", "Small", "Medium", "Large / Prominent",
             "Well-Developed", "Large (Cervicothoracic)"],
    "horn": ["Absent / Polled", "Short", "Curved", "Lyre-Shaped",
             "Semicircular", "Tightly Curled"],
    "dewlap": ["Minimal", "Small", "Moderate", "Moderate & Pendulous",
               "Large & Pendulous"],
    "size": ["Small / Compact", "Small-Medium", "Medium", "Medium-Large", "Large"],
    "ear": ["Small", "Medium", "Long & Pendulous", "Drooping"],
    "face": ["Short", "Medium", "Long", "Broad"],
    "coat": ["Light", "Spotted", "Grey", "Brown", "Black"],
}


def humanize_trait(field, token):
    """Map a snake_case trait token to a display string."""
    if token is None:
        return "Unknown"
    table = TRAIT_DISPLAY.get(field, {})
    if token in table:
        return table[token]
    generic = TRAIT_GENERIC_LABELS.get(field, [])
    if token.startswith("class_"):
        try:
            idx = int(token.split("_", 1)[1])
        except (IndexError, ValueError):
            return token
        if 0 <= idx < len(generic):
            return generic[idx]
        return f"{field.title()} class {idx}"
    return token.replace("_", " ").title()
