"""
===============================================================================
Aesthify Utilities Initialization
===============================================================================
Exposes core utility modules for the backend:
- Aesthetic scoring principles
- Object detection pipelines
- Image processing utilities
- Main scoring pipeline
"""

# Aesthetic Scoring Functions
from .aesthetic_scoring import (
    calculate_styling_balance,
    calculate_color_balance,
    calculate_roughness_balance,
    calculate_proportion_score,
    calculate_symmetry_score,
    calculate_simplicity_score,
    calculate_harmony_score,
    calculate_contrast_score,
    calculate_unity_score,
    estimate_simplicity_from_roughness,
    group_by_similarity,
    group_by_proximity,
    group_by_closure,
    group_by_continuation,
    group_by_figure_ground,
    compute_normalized_centroid,
)

# Detection Pipeline Functions
from .detection_pipeline import (
    run_yolo_detection,
    multi_model_detect,
    non_max_suppression,
    compute_iou,
    resolve_label_conflicts
)

# Image Processing Utilities
from .image_utils import (
    decode_image,
    encode_image_with_labels,
    edge_detect,
    identify_elements_and_body,
    extract_color_and_size_information,
    detect_and_crop_objects
)

# Main Pipeline Functions
from .main_pipeline import (
    process_image_with_bboxes,
    process_top,
    cluster_contours_by_kmeans
)

# Configuration Variables
from .config import (
    ROBOFLOW_API_KEY,
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
    NMS_IOU_THRESHOLD,
    DETECTION_MODELS,
    MIN_AREA_RATIO,
    GESTALT_PROXIMITY_THRESHOLD,
    GESTALT_SIMILARITY_THRESHOLD,
    RESULTS_DUMP,
    SURVEY_DATA_PATH,
    EVALUATION_DATA_PATH,
    MAX_CONTENT_LENGTH_MB
)

# Exports
__all__ = [
    # aesthetic_scoring
    "calculate_styling_balance", "calculate_color_balance", "calculate_roughness_balance",
    "calculate_proportion_score", "calculate_symmetry_score", "calculate_simplicity_score",
    "calculate_harmony_score", "calculate_contrast_score", "calculate_unity_score",
    "estimate_simplicity_from_roughness", "group_by_similarity", "group_by_proximity",
    "group_by_closure", "group_by_continuation", "group_by_figure_ground",
    "compute_normalized_centroid",

    # detection_pipeline
    "run_yolo_detection", "multi_model_detect", "non_max_suppression",
    "compute_iou", "resolve_label_conflicts",

    # image_utils
    "decode_image", "encode_image_with_labels", "edge_detect",
    "identify_elements_and_body", "extract_color_and_size_information", "detect_and_crop_objects",

    # main_pipeline
    "process_image_with_bboxes", "process_top", "cluster_contours_by_kmeans"
]