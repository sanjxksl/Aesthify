import base64
import numpy as np
import cv2

from utils.detection_pipeline import run_yolo_detection
from utils.config import YOLO_CONFIDENCE_THRESHOLD, MIN_AREA_RATIO


def edge_detect(image):
    """
    Applies adaptive Canny edge detection followed by morphological cleanup.

    Steps:
    - Converts image to grayscale.
    - Applies Gaussian blur to smooth noise.
    - Computes adaptive thresholds using image median.
    - Applies Canny edge detector with adaptive thresholds.
    - Uses dilation and erosion to clean up the edge map.

    Parameters:
    -----------
    image : np.ndarray
        Input image (BGR format).

    Returns:
    --------
    edges : np.ndarray
        Binary edge map with refined contours.
    """
    ksize = 3
    kernel = np.ones((ksize, ksize), np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smoothen using Gaussian blur
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)

    # Use median of blurred image for adaptive thresholds
    v = np.median(gray_blurred)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))

    # Canny edge detection
    edges = cv2.Canny(gray_blurred, lower, upper)

    # Morphological cleanup to fill gaps and refine structure
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges

def decode_image(image_data: str) -> np.ndarray:
    """
    Decode a base64-encoded image URI into a BGR OpenCV image.

    Args:
        image_data (str): Base64 image string (data URI format).

    Returns:
        np.ndarray: Decoded OpenCV image (BGR).

    Raises:
        ValueError: If the decoding fails or the image is invalid.
    """
    try:
        # Strip base64 prefix and decode into bytes
        b64  = image_data.split(',', 1)[1]
        data = base64.b64decode(b64)

        # Convert byte stream into OpenCV image
        arr  = np.frombuffer(data, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("decode_image: could not decode image")

        return img

    except Exception as e:
        raise ValueError(f"decode_image error: {e}")

def encode_image_with_labels(image, bboxes, labels):
    vis_img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x1, y1, x2, y2), label in zip(bboxes, labels):
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, label, (x1, y1 - 10), font, 0.5, (255, 0, 0), 2)

    _, buffer = cv2.imencode('.jpg', vis_img)
    encoded = base64.b64encode(buffer).decode('utf-8')  # No prefix added
    return encoded

def detect_and_crop_objects(image: np.ndarray,
                            conf_thresh: float = 0.3,
                            visualize: bool = False):
    """
    Runs YOLOv8 on an image to detect objects and extract cropped segments,
    along with a dynamic body bounding box created from the convex hull
    of all detected object coordinates.

    Parameters:
    -----------
    image : np.ndarray
        Input image in BGR format (as used by OpenCV).
    conf_thresh : float
        Minimum confidence score for YOLO detections to be considered.
    visualize : bool
        If True, overlays the detected boxes and dynamic body on the image.

    Returns:
    --------
    segments : list of dict
        Each dict contains:
            - "crop": Cropped image region (np.ndarray)
            - "centroid": Tuple (cx, cy) normalized to image dimensions
            - "bbox": Bounding box tuple (x1, y1, x2, y2)
            - "conf": Detection confidence (float)
            - "cls": Class ID (int)

    body_bbox : tuple
        Dynamic bounding box for the "body" computed from convex hull,
        formatted as (x1, y1, x2, y2). Falls back to full image if no objects.
    """
    results = run_yolo_detection(image)
    h, w = image.shape[:2]
    segments = []

    if results.boxes is None or len(results.boxes) == 0:
        print("[WARN] No objects detected by YOLO.")
        return [], (0, 0, w, h)  # fallback: full image

    all_points = []

    for box in results.boxes:
        if float(box.conf) < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = image[y1:y2, x1:x2].copy()
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        segments.append({
            "crop": crop,
            "centroid": (cx, cy),
            "bbox": (x1, y1, x2, y2),
            "conf": float(box.conf),
            "cls": int(box.cls)
        })
        all_points.extend([[x1, y1], [x2, y2]])

    # Construct dynamic body from convex hull around elements
    all_points = np.array(all_points)
    if len(all_points) >= 3:
        hull = cv2.convexHull(all_points)
        x, y, w_body, h_body = cv2.boundingRect(hull)
        body_bbox = (x, y, x + w_body, y + h_body)
    else:
        body_bbox = (0, 0, w, h)

    # Optional overlay visualization
    if visualize:
        vis_img = image.copy()
        for seg in segments:
            x1, y1, x2, y2 = seg["bbox"]
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        x1, y1, x2, y2 = body_bbox
        if visualize:
            from matplotlib import pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.title("Detection Overlay")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    return segments, body_bbox

def identify_elements_and_body(image, fallback_boxes=None, min_area_ratio = MIN_AREA_RATIO):
    """
    Identifies meaningful element contours and body contour using edge map.
    Falls back to bounding boxes (e.g., YOLO) if contours are insufficient.

    Parameters:
    -----------
    image : np.ndarray
        Input image (BGR format).
    fallback_boxes : list of dict (optional)
        List of dicts with "bbox" key to use as fallback if no contours are found.
    min_area_ratio : float
        Minimum contour area relative to image size for it to be considered valid.

    Returns:
    --------
    element_contours : list of np.ndarray
        List of contours representing detected elements.
    body_contour : np.ndarray or None
        The largest contour treated as "body" of the layout.
    eroded_mask : np.ndarray
        Binary mask for body after refinement.
    """
    edges = edge_detect(image)
    height, width = image.shape[:2]
    min_area = (height * width) * min_area_ratio
    element_contours = []

    # Find external contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    body_contour = None
    max_area = 0

    # Classify contours based on area
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if area > max_area:
            if body_contour is not None:
                element_contours.append(body_contour)
            body_contour = cnt
            max_area = area
        else:
            element_contours.append(cnt)

    # Fallback mechanism using detection boxes if no good contours
    if body_contour is None or len(element_contours) == 0:
        print("[WARN] No usable contours from edge map. Falling back to YOLO boxes.")
        element_contours = []
        all_points = []

        for box in fallback_boxes or []:
            x1, y1, x2, y2 = box["bbox"]
            cnt = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]])
            element_contours.append(cnt)
            all_points.extend(cnt[:, 0])

        if all_points:
            hull = cv2.convexHull(np.array(all_points))
            body_contour = hull
        else:
            print("[ERROR] Fallback boxes were also empty.")
            return [], None, None

    # Create mask from body contour
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [body_contour], -1, 255, thickness=cv2.FILLED)

    # Refine mask using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=4)

    # Final cleanup of body mask
    final_contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if final_contours:
        body_contour = final_contours[0]

    eroded_mask = cv2.erode(mask, kernel, iterations=5)
    return element_contours, body_contour, eroded_mask

def extract_color_and_size_information(image, element_contours, body_contour):
    """
    Extracts HSV color, area, and centroid information from both element and body contours.

    Parameters:
    image : np.ndarray
        Input BGR image used for color and shape extraction.
    element_contours : list of np.ndarray
        List of element contours to analyze.
    body_contour : np.ndarray
        Contour representing the full image body or layout region.

    Returns:
    dict
        {
            "element_colors": list of (H, S, V) color tuples for each element,
            "element_sizes": list of contour areas for elements,
            "body_color": average HSV color of body region,
            "element_centroids": list of (x, y) centroid positions (normalized),
            "body_size": area of the full body region,
            "total_size": combined area of body and all elements
        }
    """
    if body_contour is None or not element_contours:
        raise ValueError("Missing body or elements")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = image.shape[:2]

    # ----- Body -----
    body_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(body_mask, [body_contour], -1, 255, thickness=cv2.FILLED)
    idx = np.where(body_mask != 0)

    body_color = (
        np.mean(hsv_image[idx[0], idx[1], 0]) * 2 if idx[0].size > 0 else 0,  # H (rescaled to 0–360)
        np.mean(hsv_image[idx[0], idx[1], 1]) if idx[0].size > 0 else 0,     # S
        np.mean(hsv_image[idx[0], idx[1], 2]) if idx[0].size > 0 else 0      # V
    )

    # ----- Elements -----
    element_colors, element_sizes, element_centroids = [], [], []

    for cnt in element_contours:
        # Create a binary mask for the current element
        mask_ = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask_, [cnt], -1, 255, thickness=cv2.FILLED)
        idx = np.where(mask_ != 0)

        # Average HSV color
        if idx[0].size > 0:
            h = np.mean(hsv_image[idx[0], idx[1], 0]) * 2
            s = np.mean(hsv_image[idx[0], idx[1], 1])
            v = np.mean(hsv_image[idx[0], idx[1], 2])
            element_colors.append((h, s, v))
        else:
            element_colors.append((0, 0, 0))

        # Contour area
        area = cv2.contourArea(cnt)
        element_sizes.append(area)

        # Centroid (normalized)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = (M["m10"] / M["m00"]) / width
            cy = (M["m01"] / M["m00"]) / height
        else:
            cx, cy = 0, 0
        element_centroids.append((cx, cy))

    body_size = cv2.contourArea(body_contour)
    total_size = body_size + sum(element_sizes)

    return {
        "element_colors": element_colors,
        "element_sizes": element_sizes,
        "body_color": body_color,
        "element_centroids": element_centroids,
        "body_size": body_size,
        "total_size": total_size
    }

def detect_and_contour(image_data: str,
                       conf_thresh: float = YOLO_CONFIDENCE_THRESHOLD
                      ) -> tuple[list[list[float]], list[np.ndarray], list[str]]:
    """
    Detect objects using YOLOv8 from a base64-encoded image string
    and construct 4-point contours from each bounding box.

    Process:
      1. Decodes image from base64 string.
      2. Performs YOLOv8 object detection.
      3. Converts boxes to rectangular contours.

    Args:
        image_data (str): Base64 image string (data URI format).
        conf_thresh (float): Minimum confidence threshold for detections.

    Returns:
        tuple:
            - bboxes (List[List[float]]): Each bounding box as [x1, y1, x2, y2].
            - element_contours (List[np.ndarray]): 4-point rectangular contours for each box.
            - detected_labels (List[str]): Class label names for each detection.
    """
    img = decode_image(image_data)
    h, w = img.shape[:2]

    # Run YOLOv8 inference
    detections = run_yolo_detection(img)
    bboxes = [list(d["bbox"]) for d in detections]
    labels = [d["label"] for d in detections]

    # Build rectangular contours (clockwise 4-point polygon)
    contours = []
    for x1, y1, x2, y2 in bboxes:
        cnt = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.int32).reshape(-1, 1, 2)
        contours.append(cnt)

    return bboxes, contours, labels
