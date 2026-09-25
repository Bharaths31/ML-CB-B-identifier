import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

SERVER_PORT = int(os.environ.get('FLASK_PORT', 5000))
CORS_ORIGINS = ['http://localhost:5173', 'http://localhost:3000']
DEBUG = os.environ.get('FLASK_DEBUG', '0') == '1'
