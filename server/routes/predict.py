import time
from flask import Blueprint, jsonify, request, current_app
from server.model_manager import get_manager
from server.services.inference import predict_image
from server.utils.logger import get_logger

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
        
    file = request.files['image']
    image_bytes = file.read()
    model_name = request.form.get("model", "")
    filename = request.form.get("filename", "unknown")
    
    logger = get_logger()
    if logger:
        logger.info(f"Model selected: {model_name}")

    socketio = current_app.extensions.get('socketio')
    
    if socketio:
        socketio.emit('prediction_started', {
            'filename': filename, 
            'model': model_name,
            'timestamp': time.time()
        }, namespace='/ws')

    try:
        manager = get_manager()
        result = predict_image(manager, image_bytes, model_name, socketio=socketio)
        result["filename"] = filename
        
        # We can pass `devMode` from the frontend, but here we just return all data 
        # and let frontend handle filtering.
        
        return jsonify(result)
    except Exception as e:
        if socketio:
            socketio.emit('prediction_error', {'error': str(e), 'filename': filename}, namespace='/ws')
        return jsonify({"error": str(e)}), 500
