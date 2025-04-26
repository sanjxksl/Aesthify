"""
===============================================================================
Aesthify Object Detection Pipeline
===============================================================================

Utilities to:
- Run YOLOv8 object detection
- Run multi-model Roboflow detection
- Perform Non-Maximum Suppression (NMS)
- Resolve conflicting detections across models

Optimized for production usage with preloaded models.
"""

# ========== IMPORTS ==========

import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from utils.config import (
    ROBOFLOW_API_KEY,
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
    NMS_IOU_THRESHOLD,
    DETECTION_MODELS
)

# ========== GLOBAL SETUP ==========

# Set API key for Roboflow environment
os.environ["ROBOFLOW_API_KEY"] = ROBOFLOW_API_KEY

# Load YOLOv8 model once globally for efficiency
yolo_model = YOLO(YOLO_MODEL_PATH)

# ========== FUNCTIONS ==========

def run_yolo_detection(image):
    """
    Run YOLOv8 object detection on a given image.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        list of dict: List of detections with bbox, label, and confidence.
    """
    # Perform detection with pre-loaded YOLO model
    results = yolo_model.predict(image, imgsz=640, conf=YOLO_CONFIDENCE_THRESHOLD)
    detections = []

    # Iterate through each result object
    for r in results:
        for box in r.boxes.data:
            # Extract box coordinates, confidence, and class
            x1, y1, x2, y2, conf, cls = box.tolist()
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "label": r.names[int(cls)]
            })

    # Return formatted detection list
    return detections

def multi_model_detect(image: np.ndarray) -> list[dict]:
    """
    Run object detection using multiple Roboflow models.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        list of dict: Consolidated detection results across models.
    """
    detections = []
    model_ids = DETECTION_MODELS

    # Run inference on each Roboflow model
    for model_id in model_ids:
        try:
            from inference import get_model
            import supervision as sv

            # Load Roboflow model by ID
            model = get_model(model_id=model_id)
            results = model.infer(image)

            # Support multiple outputs (rare)
            if isinstance(results, list):
                results = results[0]

            # Convert raw results to supervision format
            supervision_detections = sv.Detections.from_inference(results)

            # Try to get label mapping (if any)
            class_names = getattr(results.predictions, "class_map", {})

            # Parse detections
            for xyxy, conf, class_id in zip(
                supervision_detections.xyxy,
                supervision_detections.confidence,
                supervision_detections.class_id
            ):
                x1, y1, x2, y2 = map(int, xyxy)
                # Handle possible missing label map
                if isinstance(class_names, dict):
                    label = class_names.get(str(class_id), f"class_{class_id}")
                elif isinstance(class_names, list):
                    label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                else:
                    label = f"class_{class_id}"

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "label": label,
                    "conf": float(conf)
                })

        except Exception as e:
            print(f"[ERROR] Roboflow model {model_id} failed: {e}")
            continue

    return detections

def non_max_suppression(detections, iou_threshold=NMS_IOU_THRESHOLD):
    """
    Perform Non-Maximum Suppression (NMS) to reduce redundant detections.

    Args:
        detections (list of dict): Raw detection list.
        iou_threshold (float): IOU threshold for suppression.

    Returns:
        list of dict: Pruned detection list.
    """
    if not detections:
        return []

    # Unpack boxes and scores
    boxes = np.array([det["bbox"] for det in detections])
    scores = np.array([det["conf"] for det in detections])

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # Sort descending

    keep = []

    # Greedily select best boxes
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute overlaps
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1 + 1)
        inter_h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = inter_w * inter_h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep only boxes with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return [detections[i] for i in keep]

def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA (tuple): (x1, y1, x2, y2)
        boxB (tuple): (x1, y1, x2, y2)

    Returns:
        float: IoU score.
    """
    # Compute intersection coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection and union areas
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute IoU with small epsilon to avoid division by zero
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

def resolve_label_conflicts(detections, iou_threshold=NMS_IOU_THRESHOLD):
    """
    Resolve overlapping detections with conflicting class labels.

    Args:
        detections (list of dict): Raw detections.
        iou_threshold (float): IOU threshold for merging.

    Returns:
        list of dict: Final consistent detections.
    """
    import torch
    from torchvision.ops import nms

    # Convert to PyTorch tensors
    boxes = [d["bbox"] for d in detections]
    scores = [d["conf"] for d in detections]

    if not boxes:
        return []

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    indices = nms(boxes_tensor, scores_tensor, iou_threshold)
    final = [detections[i] for i in indices]

    merged = []

    # Filter boxes that disagree in label
    for i in range(len(final)):
        keep = True
        for j in range(len(final)):
            if i == j:
                continue
            box_i = final[i]["bbox"]
            box_j = final[j]["bbox"]
            iou = compute_iou(box_i, box_j)
            if iou > 0.5 and final[i]["label"] != final[j]["label"]:
                keep = final[i]["conf"] > final[j]["conf"]
        if keep:
            merged.append(final[i])

    return merged