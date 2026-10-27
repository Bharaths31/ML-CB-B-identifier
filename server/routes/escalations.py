"""Veterinary expert escalation queue + active-learning loop endpoints."""

from __future__ import annotations

import base64
import json
from flask import Blueprint, jsonify, request, current_app

from server.services.escalation_db import get_escalation_db
from server.utils.logger import get_logger

escalations_bp = Blueprint("escalations", __name__)

_JSON_FIELDS = ("ood_details", "top3", "quality", "top5_breeds")
_NUMERIC_FIELDS = ("gps_lat", "gps_lng", "confidence", "confidence_level")
_MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _validate_top3(value):
    if value is None:
        return None
    if not isinstance(value, list):
        return "top3 must be a JSON array"
    for item in value:
        if not isinstance(item, dict) or not (item.get("breed") or item.get("name")):
            return "each top3 item must contain a breed name"
    return None


def _validate_image(value):
    if not value:
        return None
    if not isinstance(value, str):
        return "image_b64 must be a string"
    payload = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception:
        return "image_b64 is not valid base64"
    if not raw:
        return "image_b64 is empty"
    if len(raw) > _MAX_IMAGE_BYTES:
        return "image_b64 exceeds the 8 MB limit"
    return None


def _coerce_payload(raw):
    payload = dict(raw or {})
    for key in _JSON_FIELDS:
        value = payload.get(key)
        if isinstance(value, str):
            try:
                payload[key] = json.loads(value)
            except Exception:
                pass
    for key in _NUMERIC_FIELDS:
        if key in payload and payload[key] not in (None, ""):
            try:
                payload[key] = float(payload[key]) if key in ("gps_lat", "gps_lng", "confidence") else int(payload[key])
            except (TypeError, ValueError):
                payload.pop(key, None)
    for key in ("is_ood", "promote_to_dataset"):
        if key in payload and isinstance(payload[key], str):
            payload[key] = payload[key].lower() in ("1", "true", "yes", "on")
    return payload


def _payload_from_request():
    if request.is_json:
        return _coerce_payload(request.get_json(silent=True) or {})

    form = request.form.to_dict()
    # Some clients send one JSON blob under `payload`.
    if "payload" in form:
        try:
            parsed = _coerce_payload(json.loads(form.pop("payload")))
        except Exception:
            parsed = {}
        if parsed:
            form.update({k: v for k, v in parsed.items() if k not in form or k == "filename"})
    payload = _coerce_payload(form)

    if "image" in request.files:
        upload = request.files["image"]
        raw = upload.read()
        if raw:
            payload["image_b64"] = (
                "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
            )
            payload.setdefault("filename", upload.filename or "escalation.jpg")
    return payload


@escalations_bp.route("/api/escalations", methods=["POST"])
def create_escalation():
    payload = _payload_from_request()
    if not payload.get("predicted_species") and not payload.get("species"):
        return jsonify({"error": "predicted_species is required"}), 400
    if not payload.get("predicted_breed") and not payload.get("top_breed"):
        return jsonify({"error": "predicted_breed is required"}), 400

    top3_error = _validate_top3(payload.get("top3") or payload.get("top5_breeds"))
    if top3_error:
        return jsonify({"error": top3_error}), 400
    image_error = _validate_image(payload.get("image_b64"))
    if image_error:
        return jsonify({"error": image_error}), 400

    try:
        record = get_escalation_db().create(payload)
    except Exception as exc:
        logger = get_logger()
        if logger:
            logger.error(f"Escalation insert failed: {exc}")
        return jsonify({"error": str(exc)}), 500

    socketio = current_app.extensions.get("socketio")
    if socketio:
        socketio.emit(
            "escalation_created",
            {
                "id": record["id"],
                "status": record["status"],
                "confidence_level": record.get("confidence_level"),
                "predicted_species": record.get("predicted_species"),
                "predicted_breed": record.get("predicted_breed"),
                "created_at": record.get("created_at"),
            },
            namespace="/ws",
        )

    logger = get_logger()
    if logger:
        logger.info(
            f"Escalation queued: {record['id']} "
            f"({record.get('predicted_species')} / {record.get('predicted_breed')}, "
            f"level {record.get('confidence_level')})"
        )
    return jsonify({"status": "queued", "escalation": record}), 201


@escalations_bp.route("/api/escalations", methods=["GET"])
def list_escalations():
    status = (request.args.get("status") or "all").lower()
    if status not in ("all", "pending", "verified", "rejected"):
        return jsonify({"error": "status must be all|pending|verified|rejected"}), 400
    try:
        limit = max(1, min(int(request.args.get("limit", 200)), 1000))
    except ValueError:
        limit = 200
    data = get_escalation_db().list(status=status, limit=limit)
    data["status"] = status
    return jsonify(data)


@escalations_bp.route("/api/escalations/<escalation_id>", methods=["GET"])
def get_escalation(escalation_id):
    record = get_escalation_db().get(escalation_id)
    if not record:
        return jsonify({"error": "Escalation not found"}), 404
    return jsonify({"escalation": record})


@escalations_bp.route("/api/escalations/<escalation_id>/resolve", methods=["POST"])
def resolve_escalation(escalation_id):
    payload = _payload_from_request()
    status = (payload.get("status") or "verified").lower()
    if status not in ("verified", "rejected"):
        return jsonify({"error": "status must be 'verified' or 'rejected'"}), 400
    if status == "verified":
        species = str(payload.get("expert_species") or "").lower()
        if species not in ("cattle", "buffalo"):
            return jsonify({"error": "expert_species must be cattle or buffalo"}), 400
        if not payload.get("expert_breed"):
            return jsonify({"error": "expert_breed is required to verify"}), 400

    try:
        record = get_escalation_db().resolve(escalation_id, payload)
    except KeyError:
        return jsonify({"error": "Escalation not found"}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger = get_logger()
        if logger:
            logger.error(f"Escalation resolve failed: {exc}")
        return jsonify({"error": str(exc)}), 500

    socketio = current_app.extensions.get("socketio")
    if socketio:
        socketio.emit(
            "escalation_resolved",
            {
                "id": record.get("id"),
                "status": record.get("status"),
                "expert_species": record.get("expert_species"),
                "expert_breed": record.get("expert_breed"),
                "active_learning": record.get("active_learning"),
            },
            namespace="/ws",
        )
    return jsonify({"status": "ok", "escalation": record})
