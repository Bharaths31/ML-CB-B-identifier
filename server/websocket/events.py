import time
from flask_socketio import Namespace, emit
from server.model_manager import get_manager
from server.utils.logger import get_logger
from server.shared_config import get_shared_config

class SystemEventsNamespace(Namespace):
    def on_connect(self):
        logger = get_logger()
        if logger:
            logger.info("Client connected to WebSocket")
            
        manager = get_manager()
        db = get_shared_config()
        emit('server_status', {
            'models_count': len(manager.models) if manager else 0,
            'device': manager.device.type if manager else "cpu",
            'uptime': time.time() - logger.start_time if logger else 0,
            'active_model': manager.get_active_model_name() if manager else None,
            'config': db.get_all(),
        })

    def on_disconnect(self):
        logger = get_logger()
        if logger:
            logger.info("Client disconnected from WebSocket")

    def on_request_status(self, data=None):
        manager = get_manager()
        logger = get_logger()
        db = get_shared_config()
        emit('server_status', {
            'models_count': len(manager.models) if manager else 0,
            'device': manager.device.type if manager else "cpu",
            'uptime': time.time() - logger.start_time if logger else 0,
            'active_model': manager.get_active_model_name() if manager else None,
            'config': db.get_all(),
        })
