"""
===============================================================================
Aesthify Evaluation API Endpoint
===============================================================================
Provides an API endpoint to score uploaded images aesthetically.

Accepts:
- POST request with 'image_data' field (Base64-encoded image).

Returns:
- JSON containing individual metric scores and an average aesthetic value.
"""

from flask import request, jsonify
import threading
import pandas as pd
import os

from routes import routes  # Blueprint instance
from utils.main_pipeline import process_top  # Core image processing function
from utils.config import RESULTS_DUMP  # Path to results storage

# ========== Helper Function ==========

def save_result_to_excel(result, file_path):
    """
    Save a single evaluation result to an Excel file asynchronously.

    Args:
        result (dict): Evaluation result dictionary.
        file_path (str): Path to the Excel file where results will be stored.
    """
    try:
        image_id = 1
        if os.path.isfile(file_path):
            df = pd.read_excel(file_path)
            if not df.empty:
                last_id = df["Image_ID"].iloc[-1]
                image_id = last_id + 1
        result["Image_ID"] = image_id

        if os.path.isfile(file_path):
            df = pd.concat([df, pd.DataFrame.from_records([result])], ignore_index=True)
        else:
            df = pd.DataFrame.from_records([result])

        df.to_excel(file_path, index=False)
        print("[DEBUG] Successfully saved result to Excel.")

    except Exception as e:
        print(f"[ERROR] Failed to save Excel: {e}")

# ========== API Endpoint ==========

@routes.route('/evaluate', methods=['POST'])
def evaluate():
    """
    POST /evaluate
    Accepts an uploaded base64-encoded image, processes it, returns aesthetic scores,
    and saves the results asynchronously.

    Request:
        - FormData or JSON: { "image_data": "..." }

    Response:
        - JSON: { <score metrics>, "avg_score": float }
    """
    try:
        # Retrieve base64 image from form or JSON payload
        image_data = request.form.get('image_data') or request.json.get('image_data')
        if not image_data:
            return jsonify({"error": "Missing 'image_data'"}), 400

        # Basic payload size validation (maximum 32MB)
        request_size = len(image_data) / (1024 * 1024)
        if request_size > 32:
            return jsonify({"error": "Image too large"}), 413

        print("[DEBUG] Starting process_top()...")
        result = process_top(image_data)
        print("[DEBUG] Finished process_top()")

        result_copy = result.copy()

        # Save evaluation result asynchronously
        threading.Thread(target=save_result_to_excel, args=(result, RESULTS_DUMP)).start()

        # Return the result immediately
        return jsonify(result_copy)

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        return jsonify({"error": "Server error during evaluation"}), 500