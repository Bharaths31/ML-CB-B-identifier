import time
from flask_socketio import Namespace, emit
from server.model_manager import get_manager
from server.utils.logger import get_logger

class SystemEventsNamespace(Namespace):
    def on_connect(self):
        logger = get_logger()
        if logger:
            logger.info("Client connected to WebSocket")
            
        manager = get_manager()
        emit('server_status', {
            'models_count': len(manager.models),
            'device': manager.device.type,
            'uptime': time.time() - logger.start_time if logger else 0
        })

    def on_disconnect(self):
        logger = get_logger()
        if logger:
            logger.info("Client disconnected from WebSocket")

    def on_request_status(self, data):
        manager = get_manager()
        logger = get_logger()
        emit('server_status', {
            'models_count': len(manager.models),
            'device': manager.device.type,
            'uptime': time.time() - logger.start_time if logger else 0
        })
