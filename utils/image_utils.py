"""
===============================================================================
Aesthify Image Processing Utilities
===============================================================================

Provides low-level utilities for:
- Decoding base64 images
- Encoding images with labels
- Edge detection
- Element and body segmentation
- Color and size extraction
- Object detection and cropping

These utilities form the core building blocks for aesthetic scoring pipelines.
"""

# ========== IMPORTS ==========

import numpy as np
import cv2
import base64

# ========== FUNCTIONS ==========

def decode_image(base64_data: str) -> np.ndarray:
    """
    Decode a base64-encoded image string into a numpy BGR image.

    Args:
        base64_data (str): Base64-encoded image string.

    Returns:
        np.ndarray: Decoded BGR image.
    """
    # Remove potential header if present (e.g., "data:image/jpeg;base64,...")
    img_str = base64_data.split(',', 1)[1] if ',' in base64_data else base64_data
    img_data = base64.b64decode(img_str)
    nparr = np.frombuffer(img_data, np.uint8)

    # Decode image using OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def encode_image_with_labels(img: np.ndarray, bboxes: list, labels: list) -> str:
    """
    Encode a BGR image into base64 after drawing bounding boxes and labels.

    Args:
        img (np.ndarray): BGR image.
        bboxes (list): List of [x1, y1, x2, y2] bounding boxes.
        labels (list): List of label strings corresponding to boxes.

    Returns:
        str: Base64-encoded JPEG image.
    """
    # Make a copy of the input image to annotate
    img_copy = img.copy()

    # Draw each bounding box and label
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, label, (x1, max(y1 - 10, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    # Encode annotated image to JPEG
    _, buffer = cv2.imencode('.jpg', img_copy)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return img_base64

def edge_detect(img: np.ndarray) -> np.ndarray:
    """
    Perform edge detection on an image using Canny algorithm.

    Args:
        img (np.ndarray): Input BGR image.

    Returns:
        np.ndarray: Edge-detected binary image.
    """
    # Convert BGR image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    return edges

def identify_elements_and_body(edges: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Identify element contours and the main body from an edge-detected image.

    Args:
        edges (np.ndarray): Edge-detected binary image.

    Returns:
        tuple:
            - elements (list of np.ndarray): Smaller object contours.
            - body (np.ndarray): The largest detected contour (assumed main body).
    """
    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [], None

    # Sort contours by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    body = contours[0]  # Assume largest contour is body
    elements = contours[1:] if len(contours) > 1 else []

    return elements, body

def extract_color_and_size_information(
    img: np.ndarray,
    elements: list[np.ndarray],
    body: np.ndarray
) -> dict:
    """
    Extract color and size metrics from detected elements and body.

    Args:
        img (np.ndarray): Input BGR image.
        elements (list of np.ndarray): List of object contours.
        body (np.ndarray): Body contour.

    Returns:
        dict: Contains element sizes, colors, centroids, body size, and body color.
    """
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    element_sizes = []
    element_colors = []
    element_centroids = []

    # Calculate stats for each element
    for contour in elements:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

        # Compute average HSV color
        mean_color = cv2.mean(hsv, mask=mask)[:3]
        area = cv2.contourArea(contour)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = 0, 0

        element_sizes.append(area)
        element_colors.append(mean_color)
        element_centroids.append((cx / img.shape[1], cy / img.shape[0]))

    # Stats for body
    mask_body = np.zeros(img.shape[:2], dtype=np.uint8)
    if body is not None:
        cv2.drawContours(mask_body, [body], -1, 255, thickness=-1)
        body_color = cv2.mean(hsv, mask=mask_body)[:3]
        body_size = cv2.contourArea(body)
    else:
        body_color = (0, 0, 0)
        body_size = 0

    return {
        "element_sizes": element_sizes,
        "element_colors": element_colors,
        "element_centroids": element_centroids,
        "body_color": body_color,
        "body_size": body_size
    }

def detect_and_crop_objects(img: np.ndarray, contours: list[np.ndarray]) -> list[np.ndarray]:
    """
    Crop object bounding boxes from an image based on contours.

    Args:
        img (np.ndarray): Input BGR image.
        contours (list of np.ndarray): Contour list.

    Returns:
        list of np.ndarray: Cropped image regions per contour.
    """
    cropped_objects = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Extract the region of interest (ROI)
        cropped = img[y:y+h, x:x+w]
        cropped_objects.append(cropped)

    return cropped_objects