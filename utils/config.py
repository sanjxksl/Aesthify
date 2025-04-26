"""
===============================================================================
Aesthify Backend Configuration
===============================================================================
Handles environment variables, model paths, detection thresholds, and constants.
"""

import os
from dotenv import load_dotenv

# ========== Environment Setup ==========
load_dotenv()

# Roboflow API Key
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", default=None)
if ROBOFLOW_API_KEY is None:
    print("[WARN] ROBOFLOW_API_KEY not set in .env. Roboflow models may fail.")

# ========== Model Paths ==========
YOLO_MODEL_PATH = os.path.join("models", "yolov8n.pt")

# ========== Detection Parameters ==========
YOLO_CONFIDENCE_THRESHOLD = 0.25  # Minimum detection confidence
NMS_IOU_THRESHOLD = 0.5           # Non-Maximum Suppression threshold
DETECTION_MODELS = [
    "fyp-ar4lx/1",
    "living-room-hn7cw/4",
    "bedroom-gg3fh/1"
]

# ========== Scoring Thresholds ==========
MIN_AREA_RATIO = 0.0001
GESTALT_PROXIMITY_THRESHOLD = 0.1
GESTALT_SIMILARITY_THRESHOLD = 40

# ========== Data Paths ==========
RESULTS_DUMP = "evaluation_results_dump.xlsx"
SURVEY_DATA_PATH = "interior_analysis/data/survey_results.xlsx"
EVALUATION_DATA_PATH = "interior_analysis/data/evaluation_results.xlsx"

# ========== Misc Settings ==========
MAX_CONTENT_LENGTH_MB = 32  # Flask upload limit