from flask import Blueprint, jsonify, request
from server.model_manager import get_manager
from server.services.escalation_db import known_breed_map
from server.services.metadata import extract_image_metadata
from server.utils.logger import get_logger

dev_bp = Blueprint('dev', __name__)

@dev_bp.route('/api/dev-info', methods=['GET'])
def dev_info():
    manager = get_manager()
    known = known_breed_map()
    info = {
        "cattle_classes": manager.class_maps.get("cattle"),
        "buffalo_classes": manager.class_maps.get("buffalo"),
        # Union across all visible portable class maps, used by the expert
        # portal so an old-model escalation can still be verified.
        "known_cattle_classes": {name: idx for idx, name in enumerate(sorted(known["cattle"]))},
        "known_buffalo_classes": {name: idx for idx, name in enumerate(sorted(known["buffalo"]))},
    }
    return jsonify(info)

@dev_bp.route('/api/image-info', methods=['POST'])
def image_info():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    
    meta = extract_image_metadata(image_bytes)
    
    logger = get_logger()
    if logger:
        logger.info(f"Image metadata extracted: {meta.get('width', '?')}×{meta.get('height', '?')} {meta.get('format', '?')}")
        
    return jsonify(meta)

@dev_bp.route('/api/bpa/qr', methods=['POST'])
def bpa_qr():
    """Return a QR module matrix for the BPA digital card canvas.

    Rendering the matrix on the client keeps the health card fully drawable
    offline (only the matrix request needs the API).
    """
    data = request.get_json(silent=True) or {}
    text = str(data.get("data") or "").strip()
    if not text:
        return jsonify({"error": "Missing 'data' string"}), 400
    try:
        import qrcode
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=1,
            border=2,
        )
        qr.add_data(text)
        qr.make(fit=True)
        matrix = [[1 if cell else 0 for cell in row] for row in qr.get_matrix()]
        return jsonify({"matrix": matrix, "version": qr.version, "size": len(matrix)})
    except ImportError:
        return jsonify({"error": "qrcode package is not installed"}), 501
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@dev_bp.route('/api/logs', methods=['GET'])
def logs():
    logger = get_logger()
    if logger:
        content = logger.get_log_contents()
    else:
        content = "(logger not initialized)"
    # Note: text/plain is what the frontend expects
    from flask import Response
    return Response(content, mimetype='text/plain')
