#!/usr/bin/env python3
"""Build merged dataset inventory JSON and frontend JavaScript data module.

Extracts all cattle and buffalo breeds in the project, combines their dataset image
counts from the inventoried dataset runs, traits from data/breed_traits.json, and
sorts them in descending order by highest number of dataset images.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import EXPECTED_CATTLE_BREEDS, EXPECTED_BUFFALO_BREEDS
from server.services.escalation_db import known_breed_map

# Load traits
traits_path = os.path.join(PROJECT_ROOT, "data", "breed_traits.json")
with open(traits_path, "r", encoding="utf-8") as f:
    breed_traits_data = json.load(f).get("breeds", {})

# Load dataset inventory from windows data
inventory_source = os.path.join(PROJECT_ROOT, "windows data", "20260925-011758-29c2", "data.jsonl")
inv_breeds = {}
if os.path.exists(inventory_source):
    with open(inventory_source, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("event") == "dataset_inventory":
                    inv_breeds = record.get("breeds", {})
                    break
            except Exception:
                continue

# Format display name
def format_breed_name(key: str) -> str:
    name_map = {
        "gir": "Gir", "sahiwal": "Sahiwal", "red_sindhi": "Red Sindhi",
        "tharparkar": "Tharparkar", "rathi": "Rathi", "hallikar": "Hallikar",
        "amritmahal": "Amritmahal", "khillar": "Khillar", "khillari": "Khillari",
        "kangayam": "Kangayam", "bargur": "Bargur", "hariana": "Hariana",
        "kankrej": "Kankrej", "ongole": "Ongole", "deoni": "Deoni",
        "krishna_valley": "Krishna Valley", "punganur": "Punganur", "vechur": "Vechur",
        "malnad_gidda": "Malnad Gidda", "dangi": "Dangi", "nagori": "Nagori",
        "nimari": "Nimari", "kherigarh": "Kherigarh", "mewati": "Mewati",
        "motu": "Motu", "ghumsari": "Ghumsari", "ghumusari": "Ghumusari",
        "pulikulam": "Pulikulam", "umblachery": "Umblachery", "alambadi": "Alambadi",
        "kenkatha": "Kenkatha", "ponwar": "Ponwar", "bachaur": "Bachaur",
        "gaolao": "Gaolao", "malvi": "Malvi", "siri": "Siri", "ladakhi": "Ladakhi",
        "thutho": "Thutho", "badri": "Badri", "belahi": "Belahi", "bhelai": "Bhelai",
        "binjharpuri": "Binjharpuri", "dagri": "Dagri", "gangatari": "Gangatari",
        "gangatiri": "Gangatiri", "himachali_pahari": "Himachali Pahari",
        "khariar": "Khariar", "konkan_kapila": "Konkan Kapila", "kosali": "Kosali",
        "lakhimi": "Lakhimi", "poda_thirupu": "Poda Thirupu", "poda_thurpu": "Poda Thurpu",
        "purnea": "Purnea", "red_kandhari": "Red Kandhari", "shweta_kapila": "Shweta Kapila",
        "ayrshire": "Ayrshire", "brown_swiss": "Brown Swiss", "guernsey": "Guernsey",
        "holstein_friesian": "Holstein Friesian", "holstein-friesian": "Holstein Friesian",
        "jersey": "Jersey", "limousin": "Limousin", "red_dane": "Red Dane",
        "light_draught": "Light Draught", "dark_draught": "Dark Draught",
        "red_draught": "Red Draught",
        # Buffalo
        "murrah": "Murrah", "jaffarabadi": "Jaffarabadi", "nili_ravi": "Nili-Ravi",
        "banni": "Banni", "bhadawari": "Bhadawari", "bhadwari": "Bhadwari",
        "pandharpuri": "Pandharpuri", "mehsana": "Mehsana", "surti": "Surti",
        "nagpuri": "Nagpuri", "toda": "Toda", "marathwada": "Marathwada",
        "marathwadi": "Marathwadi", "chhattisgarhi": "Chhattisgarhi",
        "chilika": "Chilika", "gojri": "Gojri", "kalahandi": "Kalahandi",
        "luit": "Luit", "luit_(swamp)": "Luit (Swamp)", "other_buffalo": "Other Buffalo",
        "other": "Other Buffalo"
    }
    if key in name_map:
        return name_map[key]
    return key.replace("_", " ").replace("-", " ").title()

# Determine species
def determine_species(key: str) -> str:
    buffalo_keys = {
        "banni", "bargur_buffalo", "bhadwari", "bhadawari", "chhattisgarhi",
        "chilika", "gojri", "jaffarabadi", "kalahandi", "luit", "luit_(swamp)",
        "marathwada", "marathwadi", "mehsana", "murrah", "nagpuri", "nili_ravi",
        "other_buffalo", "other", "pandharpuri", "surti", "toda"
    }
    if key in buffalo_keys or key in EXPECTED_BUFFALO_BREEDS:
        return "buffalo"
    if key in breed_traits_data.get("buffalo", {}):
        return "buffalo"
    return "cattle"

known = known_breed_map()
canonical_cattle_set = set(EXPECTED_CATTLE_BREEDS)
canonical_buffalo_set = set(EXPECTED_BUFFALO_BREEDS)

all_breed_keys = set(canonical_cattle_set) | set(canonical_buffalo_set)
all_breed_keys |= set(known["cattle"]) | set(known["buffalo"])
all_breed_keys |= set(breed_traits_data.get("cattle", {}).keys())
all_breed_keys |= set(breed_traits_data.get("buffalo", {}).keys())
all_breed_keys |= set(inv_breeds.keys())

# Build list of breed records
breed_list = []
total_images = sum(item.get("count", 0) for item in inv_breeds.values())

for key in all_breed_keys:
    species = determine_species(key)
    count = 0
    if key in inv_breeds:
        count = inv_breeds[key].get("count", 0)
    elif key == "bhadwari" and "bhadawari" in inv_breeds:
        count = inv_breeds["bhadawari"].get("count", 0)
    elif key == "marathwadi" and "marathwada" in inv_breeds:
        count = inv_breeds["marathwada"].get("count", 0)
    elif key == "gangatiri" and "gangatari" in inv_breeds:
        count = inv_breeds["gangatari"].get("count", 0)
    elif key == "poda_thurpu" and "poda_thirupu" in inv_breeds:
        count = inv_breeds["poda_thirupu"].get("count", 0)
    elif key == "other_buffalo" and "other" in inv_breeds:
        count = inv_breeds["other"].get("count", 0)
    elif key == "ghumsari" and "ghumusari" in inv_breeds:
        count = inv_breeds["ghumusari"].get("count", 0)
    elif key == "khillari" and "khillar" in inv_breeds:
        count = inv_breeds["khillar"].get("count", 0)

    # Determine shot tier
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

    # Trait details
    trait_info = breed_traits_data.get(species, {}).get(key, {})
    if not trait_info and key in ("gangatiri", "gangatari"):
        trait_info = breed_traits_data.get(species, {}).get("gangatiri") or breed_traits_data.get(species, {}).get("gangatari", {})
    if not trait_info and key in ("bhadawari", "bhadwari"):
        trait_info = breed_traits_data.get(species, {}).get("bhadawari") or breed_traits_data.get(species, {}).get("bhadwari", {})
    if not trait_info and key in ("marathwada", "marathwadi"):
        trait_info = breed_traits_data.get(species, {}).get("marathwadi") or breed_traits_data.get(species, {}).get("marathwada", {})

    origin = trait_info.get("origin", "India")
    distinguisher = trait_info.get("key_distinguisher", "")
    hump = trait_info.get("hump", "")
    horn = trait_info.get("horn", "")
    coat = trait_info.get("coat", "")

    is_canonical = (key in canonical_cattle_set) or (key in canonical_buffalo_set)

    pct = round((count / total_images) * 100, 2) if total_images > 0 and count > 0 else 0.0

    breed_list.append({
        "key": key,
        "name": format_breed_name(key),
        "species": species,
        "count": count,
        "percentage": pct,
        "tier": tier,
        "tier_code": tier_code,
        "is_canonical": is_canonical,
        "origin": origin,
        "distinguisher": distinguisher,
        "hump": hump,
        "horn": horn,
        "coat": coat,
    })

# Sort descending by count, then alphabetically by name
breed_list.sort(key=lambda x: (-x["count"], x["name"].lower()))

# Assign rank
for idx, b in enumerate(breed_list):
    b["rank"] = idx + 1

# Calculate species totals
total_cattle_images = sum(b["count"] for b in breed_list if b["species"] == "cattle")
total_buffalo_images = sum(b["count"] for b in breed_list if b["species"] == "buffalo")

inventory_doc = {
    "dataset": "merged",
    "total_images": total_images,
    "total_breeds": len(breed_list),
    "canonical_breeds_count": len(canonical_cattle_set) + len(canonical_buffalo_set),
    "by_species": {
        "cattle": {
            "breeds": len([b for b in breed_list if b["species"] == "cattle"]),
            "canonical_breeds": len(canonical_cattle_set),
            "images": total_cattle_images,
            "percentage": round((total_cattle_images / total_images) * 100, 1) if total_images else 0,
        },
        "buffalo": {
            "breeds": len([b for b in breed_list if b["species"] == "buffalo"]),
            "canonical_breeds": len(canonical_buffalo_set),
            "images": total_buffalo_images,
            "percentage": round((total_buffalo_images / total_images) * 100, 1) if total_images else 0,
        }
    },
    "shot_tiers": {
        "many_shot": len([b for b in breed_list if b["tier_code"] == "many_shot"]),
        "medium_shot": len([b for b in breed_list if b["tier_code"] == "medium_shot"]),
        "few_shot": len([b for b in breed_list if b["tier_code"] == "few_shot"]),
        "pending": len([b for b in breed_list if b["tier_code"] == "pending"]),
    },
    "breeds": breed_list
}

# Write JSON to data/dataset_inventory/merged.json
out_dir = os.path.join(PROJECT_ROOT, "data", "dataset_inventory")
os.makedirs(out_dir, exist_ok=True)
json_out_path = os.path.join(out_dir, "merged.json")
with open(json_out_path, "w", encoding="utf-8") as f:
    json.dump(inventory_doc, f, indent=2, ensure_ascii=False)

print(f"[OK] Generated {json_out_path} with {len(breed_list)} breeds and {total_images} images.")

# Write JS module for direct import in React frontend
js_out_path = os.path.join(PROJECT_ROOT, "breed-classifier-react", "src", "data", "breedDatasetCounts.js")
with open(js_out_path, "w", encoding="utf-8") as f:
    f.write("/**\n")
    f.write(" * Cattle and Buffalo Breed Dataset Inventory.\n")
    f.write(" * Automatically generated from merged dataset metrics.\n")
    f.write(" * Arranged in descending order by number of dataset images.\n")
    f.write(" */\n\n")
    f.write("export const DATASET_SUMMARY = ")
    f.write(json.dumps({
        "totalImages": total_images,
        "totalBreeds": len(breed_list),
        "canonicalBreeds": len(canonical_cattle_set) + len(canonical_buffalo_set),
        "bySpecies": inventory_doc["by_species"],
        "shotTiers": inventory_doc["shot_tiers"],
    }, indent=2))
    f.write(";\n\n")
    f.write("export const BREED_DATASET_COUNTS = ")
    f.write(json.dumps(breed_list, indent=2, ensure_ascii=False))
    f.write(";\n\n")
    f.write("export default BREED_DATASET_COUNTS;\n")

print(f"[OK] Generated {js_out_path}.")
