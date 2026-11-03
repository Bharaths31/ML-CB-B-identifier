"""Breed knowledge base & farmer advisory endpoints.

Two routes, both pure-CPU dictionary assembly (no model, no image):

  GET  /api/breed-knowledge/<species>/<breed>
       Full knowledge entry + a neutral (gender/age-unspecified) advisory.

  POST /api/advisory-recalculate
       {"species": "Cattle", "breed": "gir",
        "gender": "Female", "age_group": "Adult"}
       -> context-specific advisory. Answers in well under 10 ms, so the UI can
       re-render instantly when the farmer toggles gender or age WITHOUT
       re-running inference.

Breed lookup is alias-tolerant ("Nili Ravi", "nili-ravi", "bhadawari", ...).
"""

import time

from flask import Blueprint, jsonify, request

from src.breed_knowledge import (get_breed_knowledge, get_farmer_advisory,
                                 list_breeds)

knowledge_bp = Blueprint("knowledge", __name__)

ADVISORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "age_group": ["Calf", "Young", "Adult"],
}


def _error(message, status=400):
    return jsonify({"error": message}), status


@knowledge_bp.route("/api/breed-knowledge/<species>/<breed>", methods=["GET"])
def breed_knowledge(species, breed):
    started = time.perf_counter()
    entry = get_breed_knowledge(species, breed)
    if entry is None:
        return _error(f"Unknown breed '{breed}' for species '{species}'", 404)

    advisory = get_farmer_advisory(species, breed)
    return jsonify({
        "species": entry["species"],
        "breed": entry["slug"],
        "knowledge": entry,
        "advisory": advisory,
        "advisory_options": ADVISORY_OPTIONS,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    })


@knowledge_bp.route("/api/breed-knowledge", methods=["GET"])
def breed_knowledge_index():
    return jsonify({
        "count": len(list_breeds()),
        "breeds": [
            {"species": species, "slug": slug, "display_name": name}
            for species, slug, name in list_breeds()
        ],
        "advisory_options": ADVISORY_OPTIONS,
    })


@knowledge_bp.route("/api/advisory-recalculate", methods=["POST"])
def advisory_recalculate():
    started = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    species = payload.get("species")
    breed = payload.get("breed") or payload.get("breed_slug")
    if not species or not breed:
        return _error("Both 'species' and 'breed' are required")

    gender = payload.get("gender")
    age_group = payload.get("age_group") or payload.get("age")

    advisory = get_farmer_advisory(species, breed, gender, age_group)
    if "error" in advisory:
        return _error(advisory["error"], 404)

    return jsonify({
        "advisory": advisory,
        "context": advisory.get("context"),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    })
