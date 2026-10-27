"""Thread-safe SQLite store for veterinary expert escalations.

The queue is intentionally file-backed (``outputs/escalations.db``) so an FLW
can keep capturing in the field while a veterinary officer reviews cases from a
different browser session or process. The schema is additive and the helper
methods return plain dictionaries ready for ``jsonify``.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from PIL import Image

from src.config import PROJECT_ROOT

DB_PATH = os.path.join(PROJECT_ROOT, "outputs", "escalations.db")
ACTIVE_LEARNING_MANIFEST = os.path.join(
    PROJECT_ROOT, "outputs", "active_learning_manifest.csv"
)
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

VALID_STATUSES = {"pending", "verified", "rejected"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS escalations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    filename TEXT,
    image_b64 TEXT,
    gps_lat REAL,
    gps_lng REAL,
    location_label TEXT,
    flw_id TEXT DEFAULT 'FLW-TN-SALEM-01',
    model_used TEXT,
    predicted_species TEXT,
    predicted_breed TEXT,
    confidence REAL,
    confidence_level INTEGER,
    is_ood INTEGER,
    ood_details_json TEXT,
    top3_json TEXT,
    quality_json TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    expert_species TEXT,
    expert_breed TEXT,
    expert_notes TEXT,
    reviewed_at TEXT,
    attention_map_b64 TEXT
);
CREATE INDEX IF NOT EXISTS idx_escalations_status
    ON escalations(status, created_at DESC);
"""

_COLUMNS = (
    "id", "created_at", "filename", "image_b64", "gps_lat", "gps_lng",
    "location_label", "flw_id", "model_used", "predicted_species",
    "predicted_breed", "confidence", "confidence_level", "is_ood",
    "ood_details_json", "top3_json", "quality_json", "status",
    "expert_species", "expert_breed", "expert_notes", "reviewed_at",
    "attention_map_b64",
)


def _utcnow():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return None


def _snake(value, fallback="unknown"):
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return cleaned or fallback


def _confidence_level(confidence, is_ood=False):
    if is_ood:
        return 1
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return 1
    if value >= 0.85:
        return 5
    if value >= 0.75:
        return 4
    if value >= 0.6:
        return 3
    if value >= 0.4:
        return 2
    return 1


_BREED_CACHE = None
_BREED_CACHE_LOCK = threading.Lock()


def known_breed_map():
    """Union of class-map names from data/splits and portable exports."""
    global _BREED_CACHE
    if _BREED_CACHE is not None:
        return _BREED_CACHE
    with _BREED_CACHE_LOCK:
        if _BREED_CACHE is not None:
            return _BREED_CACHE
        maps = {"cattle": set(), "buffalo": set()}
        split_dir = os.path.join(PROJECT_ROOT, "data", "splits")
        for species in ("cattle", "buffalo"):
            path = os.path.join(split_dir, f"{species}_classes.json")
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as fh:
                        maps[species].update(json.load(fh).keys())
                except Exception:
                    pass
        portable_root = os.path.join(PROJECT_ROOT, "outputs", "export", "portable")
        if os.path.isdir(portable_root):
            for bundle in os.listdir(portable_root):
                bundle_dir = os.path.join(portable_root, bundle)
                if not os.path.isdir(bundle_dir):
                    continue
                for species in ("cattle", "buffalo"):
                    path = os.path.join(bundle_dir, f"{species}_classes.json")
                    if os.path.exists(path):
                        try:
                            with open(path, encoding="utf-8") as fh:
                                maps[species].update(json.load(fh).keys())
                        except Exception:
                            pass
        _BREED_CACHE = {key: frozenset(value) for key, value in maps.items()}
        return _BREED_CACHE


class EscalationDB:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA)
                # Migrate older databases that predate the attention column.
                cols = {row["name"] for row in conn.execute("PRAGMA table_info(escalations)")}
                if "attention_map_b64" not in cols:
                    conn.execute("ALTER TABLE escalations ADD COLUMN attention_map_b64 TEXT")
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------ CRUD
    def create(self, payload: dict) -> dict:
        now = _utcnow()
        record = {
            "id": payload.get("id") or str(uuid.uuid4()),
            "created_at": payload.get("created_at") or now,
            "filename": payload.get("filename") or "capture.jpg",
            "image_b64": payload.get("image_b64"),
            "gps_lat": payload.get("gps_lat"),
            "gps_lng": payload.get("gps_lng"),
            "location_label": payload.get("location_label") or "",
            "flw_id": payload.get("flw_id") or "FLW-TN-SALEM-01",
            "model_used": payload.get("model_used") or payload.get("model_name"),
            "predicted_species": payload.get("predicted_species") or payload.get("species"),
            "predicted_breed": payload.get("predicted_breed") or payload.get("top_breed"),
            "confidence": _as_float(payload.get("confidence")),
            "confidence_level": _as_int(
                payload.get("confidence_level"),
                _confidence_level(payload.get("confidence"), bool(payload.get("is_ood"))),
            ),
            "is_ood": 1 if payload.get("is_ood") else 0,
            "ood_details_json": _json(payload.get("ood_details")),
            "top3_json": _json(payload.get("top3") or payload.get("top5_breeds")),
            "quality_json": _json(payload.get("quality")),
            "status": "pending",
            "expert_species": None,
            "expert_breed": None,
            "expert_notes": None,
            "reviewed_at": None,
            "attention_map_b64": payload.get("attention_map_b64"),
        }
        placeholders = ",".join("?" for _ in _COLUMNS)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    f"INSERT INTO escalations ({','.join(_COLUMNS)}) VALUES ({placeholders})",
                    tuple(record[column] for column in _COLUMNS),
                )
                conn.commit()
            finally:
                conn.close()
        return self.get(record["id"]) or record

    def get(self, escalation_id: str):
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM escalations WHERE id = ?", (escalation_id,)
                ).fetchone()
            finally:
                conn.close()
        return self._row_to_dict(row) if row else None

    def list(self, status: str = "all", limit: int = 200):
        where = ""
        params = []
        if status in VALID_STATUSES:
            where = "WHERE status = ?"
            params.append(status)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"SELECT * FROM escalations {where} ORDER BY created_at DESC LIMIT ?",
                    (*params, int(limit)),
                ).fetchall()
                counts = {
                    row["status"]: row["count"]
                    for row in conn.execute(
                        "SELECT status, COUNT(*) AS count FROM escalations GROUP BY status"
                    )
                }
            finally:
                conn.close()
        return {
            "escalations": [self._row_to_dict(row) for row in rows],
            "pending_count": counts.get("pending", 0),
            "verified_count": counts.get("verified", 0),
            "rejected_count": counts.get("rejected", 0),
            "total_count": sum(counts.values()),
        }

    def resolve(self, escalation_id: str, payload: dict) -> dict:
        record = self.get(escalation_id)
        if not record:
            raise KeyError(escalation_id)

        status = str(payload.get("status") or "verified").lower()
        if status not in {"verified", "rejected"}:
            raise ValueError("status must be 'verified' or 'rejected'")

        expert_species = payload.get("expert_species") or payload.get("species")
        expert_breed = payload.get("expert_breed") or payload.get("breed")
        notes = payload.get("expert_notes") or payload.get("notes") or ""
        promote = bool(payload.get("promote_to_dataset"))
        reviewed_at = _utcnow()

        if status == "verified":
            species_key, breed_key, error = self.validate_breed(expert_species, expert_breed)
            if error:
                raise ValueError(error)
            expert_species, expert_breed = species_key, breed_key
        else:
            expert_species = _snake(expert_species, "unknown") if expert_species else None
            expert_breed = _snake(expert_breed, "unknown") if expert_breed else None

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """UPDATE escalations
                       SET status = ?, expert_species = ?, expert_breed = ?,
                           expert_notes = ?, reviewed_at = ?
                       WHERE id = ?""",
                    (status, expert_species, expert_breed, notes, reviewed_at, escalation_id),
                )
                conn.commit()
            finally:
                conn.close()

        active_learning = None
        if status == "verified" and promote:
            active_learning = self._promote_to_dataset(
                record, expert_species, expert_breed, notes
            )

        updated = self.get(escalation_id)
        if updated is not None:
            updated["active_learning"] = active_learning
        return updated or {}

    def validate_breed(self, species, breed):
        """Return (species_key, breed_key, error) using the known class maps."""
        species_key = _snake(species, "")
        if species_key not in ("cattle", "buffalo"):
            return None, None, "expert_species must be cattle or buffalo"
        breed_key = _snake(breed, "")
        if not breed_key:
            return None, None, "expert_breed is required"
        known = known_breed_map().get(species_key, frozenset())
        if known and breed_key not in known:
            return None, None, f"'{breed}' is not a known {species_key} breed in the class maps"
        return species_key, breed_key, None

    # -------------------------------------------------------- active learning
    def _promote_to_dataset(self, record, species, breed, notes):
        image_b64 = record.get("image_b64") or ""
        if not image_b64:
            return {"promoted": False, "error": "No image payload stored"}

        species_key, breed_key, error = self.validate_breed(species, breed)
        if error:
            return {"promoted": False, "error": error}

        raw = _decode_data_url(image_b64)
        if raw is None:
            return {"promoted": False, "error": "Could not decode image_b64"}

        target_dir = os.path.join(DATA_RAW_DIR, species_key, breed_key)
        try:
            os.makedirs(target_dir, exist_ok=True)
        except Exception as exc:
            return {"promoted": False, "error": f"Could not create target folder: {exc}"}
        filename = f"escalated_{record['id']}.jpg"
        target_path = os.path.join(target_dir, filename)

        try:
            with Image.open(io.BytesIO(raw)) as image:
                image.convert("RGB").save(target_path, format="JPEG", quality=92)
        except Exception as exc:
            return {"promoted": False, "error": f"Could not write image: {exc}"}

        row = {
            "timestamp": _utcnow(),
            "escalation_id": record["id"],
            "image_path": os.path.relpath(target_path, PROJECT_ROOT),
            "species": species_key,
            "breed": breed_key,
            "confidence": record.get("confidence"),
            "confidence_level": record.get("confidence_level"),
            "model_used": record.get("model_used"),
            "flw_id": record.get("flw_id"),
            "expert_notes": notes or "",
            "source": "expert_escalation",
        }
        try:
            os.makedirs(os.path.dirname(ACTIVE_LEARNING_MANIFEST), exist_ok=True)
            header = not os.path.exists(ACTIVE_LEARNING_MANIFEST)
            with self._lock:
                with open(ACTIVE_LEARNING_MANIFEST, "a", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    if header:
                        writer.writeheader()
                    writer.writerow(row)
        except Exception as exc:
            return {
                "promoted": False,
                "image_path": os.path.relpath(target_path, PROJECT_ROOT),
                "error": f"Image saved but manifest append failed: {exc}",
            }
        return {"promoted": True, "image_path": row["image_path"], "manifest": ACTIVE_LEARNING_MANIFEST}

    @staticmethod
    def _row_to_dict(row):
        record = {key: row[key] for key in row.keys()}
        for key in ("ood_details_json", "top3_json", "quality_json"):
            value = record.get(key)
            if value:
                try:
                    record[key.replace("_json", "")] = json.loads(value)
                except Exception:
                    record[key.replace("_json", "")] = value
        record["is_ood"] = bool(record.get("is_ood"))
        return record


def _as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _decode_data_url(value):
    if not value:
        return None
    try:
        payload = value.split(",", 1)[1] if value.startswith("data:") else value
        return base64.b64decode(payload)
    except Exception:
        return None


_instance = None
_instance_lock = threading.Lock()


def get_escalation_db():
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = EscalationDB()
    return _instance
