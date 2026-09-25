#!/usr/bin/env python3
"""Populate breed_traits.json using Gemini 2.5 Flash API.

Usage:
    python scripts/populate_breed_traits.py --api-key <KEY>
    # or
    GEMINI_API_KEY=<KEY> python scripts/populate_breed_traits.py
    
Reads the existing data/breed_traits.json template, sends structured
prompts to Gemini for each breed, and writes the filled version back.
Creates a backup of the original file before overwriting.
"""

import argparse
import json
import os
import shutil
import sys
import time

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAITS_FILE = os.path.join(PROJECT_ROOT, "data", "breed_traits.json")

PROMPT_TEMPLATE = """
You are an expert in Indian cattle and buffalo breeds. For the {species} breed 
"{breed_name}", provide EXACTLY ONE short classification label for each of the 
following morphological traits. Each answer must be a brief, consistent 
descriptor (1-3 words) suitable for use as a classification label in a machine 
learning model.

Traits:
1. hump: Describe the hump (e.g., "large", "medium", "small", "absent", "well-developed")
2. horn: Describe the horns (e.g., "long curved", "short upward", "lyre-shaped", "stumpy", "absent")
3. coat: Describe the coat color/pattern (e.g., "white", "grey", "red_brown", "black", "spotted", "piebald")
4. ear: Describe the ears (e.g., "long pendulous", "medium horizontal", "small erect", "drooping")
5. dewlap: Describe the dewlap (e.g., "large pendulous", "moderate", "minimal", "absent")
6. face: Describe the face shape (e.g., "long narrow", "broad convex", "short concave", "straight profile")
7. size: Describe the body size (e.g., "large", "medium", "small", "medium-large")

Respond ONLY with a valid JSON object.
"""

FREE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite"
]
CURRENT_MODEL_IDX = 0

def call_gemini(api_key, prompt):
    global CURRENT_MODEL_IDX
    
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.2
        }
    }
    
    max_attempts = len(FREE_MODELS) * 2
    
    for attempt in range(max_attempts):
        model_name = FREE_MODELS[CURRENT_MODEL_IDX]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            if resp.status_code == 429:
                print(f"  Model {model_name} rate limited. Cycling to next model...")
                CURRENT_MODEL_IDX = (CURRENT_MODEL_IDX + 1) % len(FREE_MODELS)
                time.sleep(2)
                continue
                
            resp.raise_for_status()
            result_json = resp.json()
            text_resp = result_json["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text_resp)
        except Exception as e:
            print(f"  Attempt {attempt+1} failed with {model_name}: {e}")
            if "404" in str(e) or "400" in str(e):
                print(f"  Cycling model due to client error...")
                CURRENT_MODEL_IDX = (CURRENT_MODEL_IDX + 1) % len(FREE_MODELS)
            time.sleep(2)
            
    return None

def main():
    parser = argparse.ArgumentParser(description="Populate breed_traits.json using Gemini")
    parser.add_argument("--api-key", help="Gemini API Key")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        print("Error: --api-key or GEMINI_API_KEY environment variable required")
        sys.exit(1)
        
    if not os.path.exists(TRAITS_FILE):
        print(f"Error: {TRAITS_FILE} not found.")
        sys.exit(1)
        
    with open(TRAITS_FILE, "r") as f:
        traits_data = json.load(f)
        
    backup_file = TRAITS_FILE + ".bak"
    shutil.copy2(TRAITS_FILE, backup_file)
    print(f"Backed up {TRAITS_FILE} to {backup_file}")
    
    fields = traits_data.get("fields", [])
    
    for species in ["cattle", "buffalo"]:
        breeds = traits_data.get("breeds", {}).get(species, {})
        for i, (breed_name, breed_traits) in enumerate(breeds.items()):
            # Skip if already fully populated
            if all(breed_traits.get(f) for f in fields):
                print(f"[{species}] {breed_name}: already populated, skipping.")
                continue
                
            print(f"[{species}] {breed_name} ({i+1}/{len(breeds)})...")
            prompt = PROMPT_TEMPLATE.format(species=species, breed_name=breed_name)
            
            if args.dry_run:
                print("  (Dry run) Prompt ready.")
                continue
                
            result = call_gemini(api_key, prompt)
            if result:
                for f in fields:
                    if f in result:
                        val = str(result[f]).strip().lower()
                        # Normalizations to keep vocab small
                        if "reddish" in val: val = val.replace("reddish", "red")
                        if "brownish" in val: val = val.replace("brownish", "brown")
                        val = val.replace("-", " ").replace("  ", " ")
                        val = val.replace(" ", "_")
                        traits_data["breeds"][species][breed_name][f] = val
                print(f"  Success: {json.dumps(traits_data['breeds'][species][breed_name])}")
            else:
                print(f"  Failed to get valid response for {breed_name}")
                
            # Respect rate limits (~15 RPM on free tier)
            time.sleep(4.5)
            
    if not args.dry_run:
        with open(TRAITS_FILE, "w") as f:
            json.dump(traits_data, f, indent=2)
        print(f"\nSaved updated traits to {TRAITS_FILE}")
        
if __name__ == "__main__":
    main()
