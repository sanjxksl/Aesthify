
"""
routes/evaluation.py

Runs aesthetic scoring on an uploaded image and returns evaluation metrics.
Accepts:
    - POST JSON or FormData with 'image_data' (base64)
Returns:
    - JSON dict of all scores + average aesthetic value
"""

from flask import request, jsonify
import threading
import pandas as pd
import os

from routes import routes  # blueprint
from utils.main_pipeline import process_top
from utils.config import RESULTS_DUMP

def save_result_to_excel(result, file_path):
    """Save evaluation results to Excel file (background thread)."""
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

@routes.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        # Accept base64 image
        image_data = request.form.get('image_data') or request.json.get('image_data')
        if not image_data:
            return jsonify({"error": "Missing 'image_data'"}), 400

        # Check payload size (base64 size)
        request_size = len(image_data) / (1024 * 1024)  # in MB
        if request_size > 32:
            return jsonify({"error": "Image too large"}), 413

        print("[DEBUG] Starting process_top()...")
        result = process_top(image_data)
        print("[DEBUG] Finished process_top()")

        result_copy = result.copy()

        # Save to Excel in background thread
        threading.Thread(target=save_result_to_excel, args=(result, RESULTS_DUMP)).start()

        # Return result immediately
        return jsonify(result_copy)

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        return jsonify({"error": "Server error during evaluation"}), 500
