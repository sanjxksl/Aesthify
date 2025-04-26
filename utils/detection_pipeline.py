import numpy as np
import os
from ultralytics import YOLO
from utils.config import ROBOFLOW_API_KEY, YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD, DETECTION_MODELS

os.environ["ROBOFLOW_API_KEY"] = ROBOFLOW_API_KEY

from pathlib import Path

def load_yolo_model(YOLO_MODEL_PATH):
    try:
        _yolo_mode = YOLO(YOLO_MODEL_PATH)  # Assuming you're using YOLOv5 or similar
        return _yolo_mode
    except Exception as e:
        print(f"[ERROR] YOLO model failed to load from {YOLO_MODEL_PATH}: {e}")
        raise FileNotFoundError("YOLO model file is missing or corrupted.")

def run_yolo_detection(image, conf = YOLO_CONFIDENCE_THRESHOLD):
    """
    Runs YOLOv8 on the input image and returns detections.
    """
    _yolo_model = load_yolo_model(YOLO_MODEL_PATH)
    results = _yolo_model(image)
    detections = []
    for box in results[0].boxes:
        if float(box.conf) < conf:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "label": _yolo_model.names[int(box.cls)],
            "conf": float(box.conf),
            "source": "yolo"
        })
    return detections

def multi_model_detect(image: np.ndarray) -> list[dict]:
    """
    Run object detection on an input image using multiple Roboflow models.

    This function loads a list of Roboflow-hosted object detection models, performs inference
    on the given image using each model, and combines the detections into a unified list.

    Parameters:
    -----------
    image : np.ndarray
        A valid image array in BGR format (as returned by cv2.imread).

    Returns:
    --------
    list[dict]
        A list of dictionaries, each representing one detected object with:
            - 'bbox': Tuple[int, int, int, int] as (x1, y1, x2, y2)
            - 'label': str, object class label
            - 'conf': float, confidence score
    """

    detections = []

    # List of Roboflow model IDs (customizable for other use cases)
    model_ids = DETECTION_MODELS

    for model_id in model_ids:
        try:
            from inference import get_model
            import supervision as sv

            # Load model from Roboflow using its unique identifier
            model = get_model(model_id=model_id)

            # Run inference on the image; may return a list of results
            results = model.infer(image)
            if isinstance(results, list):
                results = results[0]  # Use the first item if multiple results

            # Convert raw inference results into supervision's Detections object
            supervision_detections = sv.Detections.from_inference(results)

            # Attempt to retrieve class label mapping (if available)
            class_names = getattr(results.predictions, "class_map", {})

            # Iterate over each detection and format it as a dictionary
            for xyxy, conf, class_id in zip(
                supervision_detections.xyxy,
                supervision_detections.confidence,
                supervision_detections.class_id
            ):
                x1, y1, x2, y2 = map(int, xyxy)

                # Try to get a human-readable label from class map
                if isinstance(class_names, dict):
                    label = class_names.get(str(class_id), f"class_{class_id}")
                elif isinstance(class_names, list):
                    label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                else:
                    label = f"class_{class_id}"  # Fallback to generic class label

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "label": label,
                    "conf": float(conf)
                })

        except Exception as e:
            print(f"[ERROR] Roboflow {model_id} failed: {e}")
            continue  # Skip model if it fails

    return detections


def non_max_suppression(detections, iou_threshold = NMS_IOU_THRESHOLD):
    """
    Performs Non-Maximum Suppression (NMS) to eliminate redundant overlapping detections.

    This function is designed to keep only the most confident detection among overlapping
    bounding boxes. It is useful when post-processing object detection results to reduce
    duplicates and improve clarity.

    Parameters:
    ----------
    detections : list of dict
        A list of detection results. Each detection dictionary must include:
            - "bbox"  : tuple of (x1, y1, x2, y2) coordinates
            - "conf"  : float, confidence score of the detection
            - "label" : str, class/category label of the detection

    iou_threshold : float, default=0.5
        Intersection-over-Union (IoU) threshold used to determine overlap.
        Detections with IoU > threshold are suppressed.

    Returns:
    -------
    list of dict
        A reduced list of detections with minimal overlap and highest confidence.
    """

    # Return immediately if no detections are present
    if len(detections) == 0:
        return []

    # Extract bounding boxes and confidence scores
    boxes = np.array([det["bbox"] for det in detections])
    scores = np.array([det["conf"] for det in detections])
    labels = [det["label"] for det in detections]

    # Split box coordinates for clarity
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute area of each bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort boxes by confidence score in descending order
    order = scores.argsort()[::-1]

    keep = []  # Indices of boxes to keep

    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Calculate coordinates of intersection boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Compute width and height of intersections
        inter_w = np.maximum(0.0, xx2 - xx1 + 1)
        inter_h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = inter_w * inter_h

        # Compute IoU
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU below the threshold
        inds = np.where(iou <= iou_threshold)[0]

        # Prepare next iteration
        order = order[inds + 1]

    # Return filtered detections
    return [detections[i] for i in keep]

def compute_iou(boxA, boxB):
        """
        Calculates Intersection over Union (IoU) between two bounding boxes.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
        return iou

def resolve_label_conflicts(detections, iou_threshold = NMS_IOU_THRESHOLD):
        """
        Applies NMS and filters boxes with overlapping regions but different labels.
        Keeps the most confident label in case of label conflict.
        """
        import torch
        from torchvision.ops import nms

        boxes = [d["bbox"] for d in detections]
        scores = [d["conf"] for d in detections]

        if not boxes:
            return []

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        indices = nms(boxes_tensor, scores_tensor, iou_threshold)
        final = [detections[i] for i in indices]

        # Optional post-NMS filtering for label disagreement in overlapping boxes
        merged = []
        for i in range(len(final)):
            keep = True
            for j in range(len(final)):
                if i == j:
                    continue
                box_i = final[i]["bbox"]
                box_j = final[j]["bbox"]
                iou = compute_iou(box_i, box_j)
                if iou > 0.5 and final[i]["label"] != final[j]["label"]:
                    # Retain the one with higher confidence
                    keep = final[i]["conf"] > final[j]["conf"]
            if keep:
                merged.append(final[i])
        return merged