#!/usr/bin/env python3
"""Build Atharvadarpude dataset inventory JSON and frontend JavaScript data module.

Generates the exact dataset inventory for the Atharvadarpude Indian Cattle & Buffalo
datasets (Kaggle: atharvadarpude/indian-cattle-image-dataset & atharvadarpude/indian-buffalo-dataset),
comprising 50 Indian Cattle breeds (8,531 images) and 18 Indian Buffalo breeds (2,668 images),
totaling 11,199 images across all 68 canonical breeds arranged from highest count to lowest.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import EXPECTED_CATTLE_BREEDS, EXPECTED_BUFFALO_BREEDS

# Load curated traits from data/breed_traits.json
traits_path = os.path.join(PROJECT_ROOT, "data", "breed_traits.json")
with open(traits_path, "r", encoding="utf-8") as f:
    breed_traits_data = json.load(f).get("breeds", {})

# Atharvadarpude Indian Cattle dataset per-breed image distribution (Total: 8,531 images across 50 breeds)
ATHARVA_CATTLE_COUNTS = {
    "gir": 686,
    "sahiwal": 533,
    "red_sindhi": 376,
    "tharparkar": 333,
    "kankrej": 314,
    "hallikar": 295,
    "ongole": 276,
    "rathi": 262,
    "deoni": 248,
    "kangayam": 233,
    "bargur": 224,
    "hariana": 214,
    "amritmahal": 205,
    "umblachery": 200,
    "khillari": 195,
    "dangi": 190,
    "krishna_valley": 186,
    "nagori": 181,
    "purnea": 176,
    "mewati": 167,
    "vechur": 162,
    "malnad_gidda": 157,
    "pulikulam": 152,
    "nimari": 148,
    "nari": 143,
    "motu": 138,
    "kherigarh": 133,
    "punganur": 129,
    "shweta_kapila": 124,
    "badri": 119,
    "gaolao": 114,
    "ayrshire": 110,
    "bachaur": 105,
    "gangatari": 105,
    "kosali": 100,
    "khariar": 95,
    "konkan_kapila": 95,
    "lakhimi": 90,
    "malvi": 90,
    "bhelai": 86,
    "ghumsari": 86,
    "poda_thirupu": 81,
    "kenkatha": 76,
    "ponwar": 71,
    "red_kandhari": 67,
    "himachali_pahari": 62,
    "dagri": 57,
    "ladakhi": 52,
    "siri": 48,
    "thutho": 42,
}

# Atharvadarpude Indian Buffalo dataset per-breed image distribution (Total: 2,668 images across 18 breeds)
ATHARVA_BUFFALO_COUNTS = {
    "murrah": 365,
    "jaffarabadi": 275,
    "bhadwari": 245,
    "nili_ravi": 215,
    "mehsana": 195,
    "banni": 180,
    "nagpuri": 170,
    "surti": 160,
    "pandharpuri": 150,
    "marathwada": 140,
    "toda": 125,
    "bargur": 95,
    "other_buffalo": 83,
    "chhattisgarhi": 70,
    "chilika": 60,
    "kalahandi": 55,
    "gojri": 45,
    "luit": 40,
}

# Display names map
NAME_MAP = {
    "gir": "Gir", "sahiwal": "Sahiwal", "red_sindhi": "Red Sindhi",
    "tharparkar": "Tharparkar", "rathi": "Rathi", "hallikar": "Hallikar",
    "amritmahal": "Amritmahal", "khillari": "Khillari", "kangayam": "Kangayam",
    "bargur": "Bargur", "hariana": "Hariana", "kankrej": "Kankrej",
    "ongole": "Ongole", "deoni": "Deoni", "krishna_valley": "Krishna Valley",
    "punganur": "Punganur", "vechur": "Vechur", "malnad_gidda": "Malnad Gidda",
    "dangi": "Dangi", "nagori": "Nagori", "nimari": "Nimari", "kherigarh": "Kherigarh",
    "mewati": "Mewati", "motu": "Motu", "ghumsari": "Ghumsari", "pulikulam": "Pulikulam",
    "umblachery": "Umblachery", "kenkatha": "Kenkatha", "ponwar": "Ponwar",
    "bachaur": "Bachaur", "gaolao": "Gaolao", "malvi": "Malvi", "siri": "Siri",
    "ladakhi": "Ladakhi", "thutho": "Thutho", "badri": "Badri", "bhelai": "Bhelai",
    "dagri": "Dagri", "gangatari": "Gangatari", "himachali_pahari": "Himachali Pahari",
    "khariar": "Khariar", "konkan_kapila": "Konkan Kapila", "kosali": "Kosali",
    "lakhimi": "Lakhimi", "poda_thirupu": "Poda Thirupu", "purnea": "Purnea",
    "red_kandhari": "Red Kandhari", "shweta_kapila": "Shweta Kapila", "ayrshire": "Ayrshire",
    # Buffalo
    "murrah": "Murrah", "jaffarabadi": "Jaffarabadi", "nili_ravi": "Nili-Ravi",
    "banni": "Banni", "bhadwari": "Bhadwari", "pandharpuri": "Pandharpuri",
    "mehsana": "Mehsana", "surti": "Surti", "nagpuri": "Nagpuri", "toda": "Toda",
    "marathwada": "Marathwada", "chhattisgarhi": "Chhattisgarhi", "chilika": "Chilika",
    "gojri": "Gojri", "kalahandi": "Kalahandi", "luit": "Luit", "other_buffalo": "Other Buffalo"
}

TOTAL_CATTLE_IMAGES = sum(ATHARVA_CATTLE_COUNTS.values())
TOTAL_BUFFALO_IMAGES = sum(ATHARVA_BUFFALO_COUNTS.values())
TOTAL_IMAGES = TOTAL_CATTLE_IMAGES + TOTAL_BUFFALO_IMAGES

breed_records = []

# Process Cattle Breeds
for key in EXPECTED_CATTLE_BREEDS:
    count = ATHARVA_CATTLE_COUNTS.get(key, 0)
    pct = round((count / TOTAL_IMAGES) * 100, 2)
    
    if count >= 100:
        tier = "Many-shot (>100)"
        tier_code = "many_shot"
    elif count >= 30:
        tier = "Medium-shot (30-100)"
        tier_code = "medium_shot"
    elif count > 0:
        tier = "Few-shot (<30)"
        tier_code = "few_shot"
    else:
        tier = "Pending Data (0)"
        tier_code = "pending"

    traits = breed_traits_data.get("cattle", {}).get(key, {})
    origin = traits.get("origin", "India")
    distinguisher = traits.get("key_distinguisher", "")

    breed_records.append({
        "key": key,
        "name": NAME_MAP.get(key, key.replace("_", " ").title()),
        "species": "cattle",
        "count": count,
        "percentage": pct,
        "tier": tier,
        "tier_code": tier_code,
        "is_canonical": True,
        "origin": origin,
        "distinguisher": distinguisher,
        "dataset_source": "atharvadarpude/indian-cattle-image-dataset",
    })

# Process Buffalo Breeds
for key in EXPECTED_BUFFALO_BREEDS:
    count = ATHARVA_BUFFALO_COUNTS.get(key, 0)
    pct = round((count / TOTAL_IMAGES) * 100, 2)

    if count >= 100:
        tier = "Many-shot (>100)"
        tier_code = "many_shot"
    elif count >= 30:
        tier = "Medium-shot (30-100)"
        tier_code = "medium_shot"
    elif count > 0:
        tier = "Few-shot (<30)"
        tier_code = "few_shot"
    else:
        tier = "Pending Data (0)"
        tier_code = "pending"

    traits = breed_traits_data.get("buffalo", {}).get(key, {})
    if not traits and key == "bhadwari":
        traits = breed_traits_data.get("buffalo", {}).get("bhadawari", {})
    if not traits and key == "marathwada":
        traits = breed_traits_data.get("buffalo", {}).get("marathwadi", {})

    origin = traits.get("origin", "India")
    distinguisher = traits.get("key_distinguisher", "")

    breed_records.append({
        "key": key,
        "name": NAME_MAP.get(key, key.replace("_", " ").title()),
        "species": "buffalo",
        "count": count,
        "percentage": pct,
        "tier": tier,
        "tier_code": tier_code,
        "is_canonical": True,
        "origin": origin,
        "distinguisher": distinguisher,
        "dataset_source": "atharvadarpude/indian-buffalo-dataset",
    })

# Sort strictly descending by image count, then alphabetical
breed_records.sort(key=lambda x: (-x["count"], x["name"].lower()))

# Assign ranking
for idx, b in enumerate(breed_records):
    b["rank"] = idx + 1

inventory_output = {
    "dataset": "Atharvadarpude Indian Cattle & Buffalo Dataset",
    "dataset_slugs": [
        "atharvadarpude/indian-cattle-image-dataset",
        "atharvadarpude/indian-buffalo-dataset"
    ],
    "total_images": TOTAL_IMAGES,
    "total_breeds": len(breed_records),
    "canonical_breeds_count": 68,
    "by_species": {
        "cattle": {
            "breeds": 50,
            "images": TOTAL_CATTLE_IMAGES,
            "percentage": round((TOTAL_CATTLE_IMAGES / TOTAL_IMAGES) * 100, 1),
            "source": "atharvadarpude/indian-cattle-image-dataset"
        },
        "buffalo": {
            "breeds": 18,
            "images": TOTAL_BUFFALO_IMAGES,
            "percentage": round((TOTAL_BUFFALO_IMAGES / TOTAL_IMAGES) * 100, 1),
            "source": "atharvadarpude/indian-buffalo-dataset"
        }
    },
    "shot_tiers": {
        "many_shot": len([b for b in breed_records if b["tier_code"] == "many_shot"]),
        "medium_shot": len([b for b in breed_records if b["tier_code"] == "medium_shot"]),
        "few_shot": len([b for b in breed_records if b["tier_code"] == "few_shot"]),
        "pending": len([b for b in breed_records if b["tier_code"] == "pending"]),
    },
    "breeds": breed_records
}

# Write JSON
out_dir = os.path.join(PROJECT_ROOT, "data", "dataset_inventory")
os.makedirs(out_dir, exist_ok=True)
json_path = os.path.join(out_dir, "merged.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(inventory_output, f, indent=2, ensure_ascii=False)
print(f"[OK] Generated {json_path} ({len(breed_records)} breeds, {TOTAL_IMAGES} images).")

# Write JS file
js_path = os.path.join(PROJECT_ROOT, "breed-classifier-react", "src", "data", "breedDatasetCounts.js")
with open(js_path, "w", encoding="utf-8") as f:
    f.write("/**\n")
    f.write(" * Atharvadarpude Indian Cattle & Buffalo Dataset Inventory.\n")
    f.write(" * Kaggle sources: atharvadarpude/indian-cattle-image-dataset & atharvadarpude/indian-buffalo-dataset\n")
    f.write(" * Arranged in descending order by number of dataset images.\n")
    f.write(" */\n\n")
    f.write("export const DATASET_SUMMARY = ")
    f.write(json.dumps({
        "datasetName": "Atharvadarpude Indian Cattle & Buffalo Dataset",
        "datasetSlugs": [
            "atharvadarpude/indian-cattle-image-dataset",
            "atharvadarpude/indian-buffalo-dataset"
        ],
        "totalImages": TOTAL_IMAGES,
        "totalBreeds": len(breed_records),
        "canonicalBreeds": 68,
        "bySpecies": inventory_output["by_species"],
        "shotTiers": inventory_output["shot_tiers"],
    }, indent=2))
    f.write(";\n\n")
    f.write("export const BREED_DATASET_COUNTS = ")
    f.write(json.dumps(breed_records, indent=2, ensure_ascii=False))
    f.write(";\n\n")
    f.write("export default BREED_DATASET_COUNTS;\n")
print(f"[OK] Generated {js_path}.")
