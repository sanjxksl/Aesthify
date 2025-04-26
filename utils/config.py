"""
utils/config.py

🔧 Centralized configuration file for Aesthtify backend.

Includes:
- Environment variables (via dotenv)
- Model paths
- Detection and optimization hyperparameters
- Thresholds and constants for aesthetic computation
"""

import os
from dotenv import load_dotenv

# ========== Environment Setup ==========
# Load environment variables from .env
load_dotenv()

# Roboflow API Key
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", default=None)
if ROBOFLOW_API_KEY is None:
    print("[WARN] ROBOFLOW_API_KEY not set in .env. Roboflow models may fail.")

# ========== Model Paths ==========
YOLO_MODEL_PATH = os.path.join("models", "yolov8n.pt")  # Path to YOLOv8 .pt file

# ========== Detection Parameters ==========
YOLO_CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence score for YOLO detection
NMS_IOU_THRESHOLD = 0.5            # IoU threshold for Non-Maximum Suppression
DETECTION_MODELS = [
    "fyp-ar4lx/1",
    "living-room-hn7cw/4",
    "bedroom-gg3fh/1"
]  # Roboflow project versions used

# ========== Image & Aesthetic Scoring Thresholds ==========
MIN_AREA_RATIO = 0.0001            # Minimum contour area ratio for valid object
GESTALT_PROXIMITY_THRESHOLD = 0.1  # Threshold for grouping by proximity
GESTALT_SIMILARITY_THRESHOLD = 40  # Threshold for grouping by color similarity (HSV)

# ========== File Paths ==========
RESULTS_DUMP = "evaluation_results_dump.xlsx"
SURVEY_DATA_PATH = "interior_analysis/data/survey_results.xlsx"
EVALUATION_DATA_PATH = "interior_analysis/data/evaluation_results.xlsx"

# ========== Misc Settings ==========
MAX_CONTENT_LENGTH_MB = 32         # Flask upload limit
