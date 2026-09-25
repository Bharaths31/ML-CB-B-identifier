"""
Flask API server for the Breed Classifier React frontend.

    python server/app.py                # Start server on port 5000
    python server/app.py --port 8000    # Custom port
    python server/app.py --debug        # Debug mode with auto-reload
"""

import os
import sys
import argparse
import time

# Ensure project root is in pythonpath
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO

from server.config import SERVER_PORT, CORS_ORIGINS, DEBUG
from server.utils.logger import init_logger
from server.model_manager import init_manager

# Import blueprints
from server.routes.models import models_bp
from server.routes.predict import predict_bp
from server.routes.dev import dev_bp
from server.routes.export import export_bp
from server.routes.config_api import config_bp

# Import websocket events
from server.websocket.events import SystemEventsNamespace

def create_app():
    # Initialize logger and manager
    logger = init_logger()
    logger.info("Starting Flask API Server...")
    manager = init_manager()

    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'secret!'
    
    # Configure CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Store socketio in app extensions so routes can access it
    app.extensions['socketio'] = socketio

    # Register blueprints
    app.register_blueprint(models_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(dev_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(config_bp)
    
    # Register websocket namespace
    socketio.on_namespace(SystemEventsNamespace('/ws'))

    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            "status": "ok",
            "device": manager.device.type,
            "models_count": len(manager.models),
            "uptime_s": time.time() - logger.start_time
        })
        
    return app, socketio

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="🐄 Breed Classifier — Flask API Server")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help=f"port to serve on (default: {SERVER_PORT})")
    parser.add_argument("--debug", action="store_true", default=DEBUG, help="enable debug mode")
    args = parser.parse_args()

    app, socketio = create_app()
    
    print("\n" + "=" * 60)
    print(f"  🐄 Breed Classifier — Flask API Server")
    print(f"  Running at http://localhost:{args.port}")
    print("=" * 60 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=args.port, debug=args.debug, use_reloader=args.debug, allow_unsafe_werkzeug=True)
