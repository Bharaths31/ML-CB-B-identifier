import time
from flask import Blueprint, jsonify, request, send_file
from server.services.odt_export import generate_odt_report
from server.utils.logger import get_logger
import io

export_bp = Blueprint('export', __name__)

@export_bp.route('/api/export-odt', methods=['POST'])
def export_odt():
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Invalid JSON payload"}), 400
            
        results = payload.get("results", {})
        odt_mode = payload.get("mode", "single")

        logger = get_logger()
        if logger:
            logger.info(f"ODT export requested ({odt_mode} mode)")

        odt_bytes = generate_odt_report(results, mode=odt_mode)
        
        return send_file(
            io.BytesIO(odt_bytes),
            mimetype="application/vnd.oasis.opendocument.text",
            as_attachment=True,
            download_name=f"breed_report_{int(time.time())}.odt"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
