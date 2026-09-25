from flask import Blueprint, jsonify, request
from server.model_manager import get_manager
from server.services.metadata import extract_image_metadata
from server.utils.logger import get_logger

dev_bp = Blueprint('dev', __name__)

@dev_bp.route('/api/dev-info', methods=['GET'])
def dev_info():
    manager = get_manager()
    info = {
        "cattle_classes": manager.class_maps.get("cattle"),
        "buffalo_classes": manager.class_maps.get("buffalo")
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
