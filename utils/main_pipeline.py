"""
===============================================================================
Aesthify Main Image Processing Pipeline
===============================================================================

Handles:
- Decoding incoming base64 images
- Performing object detection via YOLO + Roboflow
- Fusing detections intelligently
- Computing aesthetic scores based on detected elements
- Outputting annotated results
"""

# ========== IMPORTS ==========

import base64
import statistics
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from utils import *

# ========== FUNCTIONS ==========

def cluster_contours_by_kmeans(
    element_contours: list[np.ndarray],
    image_width: int,
    image_height: int,
    element_colors: list[tuple[float, float, float]]
) -> tuple[np.ndarray, int]:
    """
    Cluster element contours based on normalized centroids using KMeans.

    Args:
        element_contours (list of np.ndarray): List of element contours.
        image_width (int): Width of input image.
        image_height (int): Height of input image.
        element_colors (list of tuple): HSV color info per element.

    Returns:
        tuple:
            clusters_info (np.ndarray): Array of [contour, cluster_label, H, S, V].
            num_clusters (int): Number of optimal clusters found.
    """
    n = len(element_contours)
    if n == 0:
        return np.empty((0, 5), dtype=object), 0

    if n == 1:
        # Single element fallback — one cluster
        single = np.array([[element_contours[0], 0, *element_colors[0]]], dtype=object)
        return single, 1

    # Normalize contour centroids
    centroids = np.array([
        compute_normalized_centroid(cnt, image_width, image_height)
        for cnt in element_contours
    ], dtype=np.float32)

    best_score = -1.0
    best_labels = None
    max_k = min(20, n - 1)

    # Try clustering for various k and choose best via silhouette score
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=1, max_iter=100)
        labels = km.fit_predict(centroids)
        score = silhouette_score(centroids, labels) if n > 2 else 1.0

        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
        best_k = 1

    # Combine contour data, cluster labels, and colors
    contours_arr = np.empty(n, dtype=object)
    contours_arr[:] = element_contours
    colors_arr = np.array(element_colors, dtype=object)

    clusters_info = np.concatenate([
        contours_arr.reshape(-1, 1),
        best_labels.reshape(-1, 1),
        colors_arr.reshape(-1, 3)
    ], axis=1)

    return clusters_info, best_k

def process_image_with_bboxes(
    image_data: str,
    bboxes: list[list[float]],
    labels: list[str],
    axis: str = "vertical",
    Nr: int = 0,
    Dmin: float = 1.0,
    Dmax: float = 10.0
) -> dict:
    """
    Compute aesthetic scores for an image given bounding boxes and labels.

    Args:
        image_data (str): Base64-encoded image string.
        bboxes (list of list): Bounding boxes [x1, y1, x2, y2].
        labels (list): Class labels per box.
        axis (str): Axis for symmetry evaluation.
        Nr (int): Simplicity reference count.
        Dmin (float): Minimum simplicity degree.
        Dmax (float): Maximum simplicity degree.

    Returns:
        dict: Dictionary of aesthetic scores.
    """
    # --- Decode base64 image ---
    img_str = image_data.split(',', 1)[1]
    data = base64.b64decode(img_str)
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image data"}

    h, w = img.shape[:2]

    # --- Separate body and element bounding boxes ---
    if bboxes:
        areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in bboxes]
        max_idx = int(np.argmax(areas))
        body_bbox = bboxes[max_idx]
        element_bboxes = [b for i, b in enumerate(bboxes) if i != max_idx]
    else:
        # Default to entire image as body if no bboxes
        body_bbox = [0, 0, w, h]
        element_bboxes = []

    # --- Convert boxes into contours ---
    x1, y1, x2, y2 = map(int, body_bbox)
    body_cnt = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32).reshape(-1, 1, 2)
    body_size = cv2.contourArea(body_cnt)

    el_contours = []
    for (ex1, ey1, ex2, ey2) in element_bboxes:
        cnt = np.array([[ex1, ey1], [ex2, ey1], [ex2, ey2], [ex1, ey2]], dtype=np.int32).reshape(-1, 1, 2)
        el_contours.append(cnt)

    # --- Extract sizes, colors, and centroids ---
    info = extract_color_and_size_information(img, el_contours, body_cnt)
    el_colors = info["element_colors"]
    el_sizes = info["element_sizes"]
    body_color = info["body_color"]
    el_centroids = info["element_centroids"]
    body_size = info["body_size"]

    total_size = body_size + sum(el_sizes)

    # --- Calculate Body Centroid ---
    M = cv2.moments(body_cnt)
    if M['m00'] > 0:
        body_centroid = ((M['m10'] / M['m00']) / w, (M['m01'] / M['m00']) / h)
    else:
        body_centroid = (0.5, 0.5)

    # --- 1. Balance Score ---
    edges = edge_detect(img)
    s_balance = calculate_styling_balance(el_sizes, el_centroids, body_size, body_centroid, total_size)
    c_balance = calculate_color_balance(el_colors, body_color, el_centroids, el_sizes, body_size, body_centroid, total_size)
    t_balance = calculate_roughness_balance(el_contours, edges, el_centroids, el_sizes, body_cnt, body_centroid)
    balance_score = (s_balance + c_balance + t_balance) / 3

    # --- 2. Proportion Score ---
    elems_info = []
    for cnt, (cx, cy), sz in zip(el_contours, el_centroids, el_sizes):
        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        elems_info.append({
            "width": w_, "height": h_,
            "centroid_x": cx, "centroid_y": cy,
            "size": sz
        })

    body_info = {
        "width": w, "height": h,
        "centroid_x": body_centroid[0], "centroid_y": body_centroid[1],
        "size": body_size
    }
    proportion_score = calculate_proportion_score(elems_info, body_info)

    # --- 3. Symmetry Score ---
    symmetry_score = calculate_symmetry_score(el_contours, img, axis)

    # --- 4. Simplicity Score ---
    try:
        edges = edge_detect(img)
        el_degrees = estimate_simplicity_from_roughness(el_contours, edges, el_sizes) if el_contours else []
        body_degree = estimate_simplicity_from_roughness([body_cnt], edges, [body_size])[0] if body_cnt is not None else Dmin

        simplicity_score = calculate_simplicity_score(
            el_sizes, el_degrees, body_size, body_degree,
            reference_count=Nr, D_min=Dmin, D_max=Dmax,
            weights=(0.5, 0.5)
        )
    except Exception as e:
        print("[simplicity] ERROR:", e)
        simplicity_score = 0.0

    # --- 5. Harmony and Contrast Scores ---
    harmony_score = calculate_harmony_score(el_colors, el_sizes, body_color, body_size)
    contrast_score = calculate_contrast_score(el_colors, el_sizes, body_color, body_size)

    # --- 6. Unity Score via Clustering ---
    clusters_info, num_clusters = cluster_contours_by_kmeans(el_contours, w, h, el_colors)

    if clusters_info.shape[0] == 0:
        # No valid clusters
        unity_score = 0.0
    else:
        # Compute Gestalt grouping scores
        similarity_counts = [
            group_by_similarity(clusters_info[clusters_info[:, 1] == i, 2:].tolist())
            for i in range(num_clusters)
        ]
        proximity_counts = [
            group_by_proximity([
                compute_normalized_centroid(cnt, w, h)
                for cnt in clusters_info[clusters_info[:, 1] == i, 0]
            ])
            for i in range(num_clusters)
        ]
        closure_counts = [
            group_by_closure(clusters_info[clusters_info[:, 1] == i, 0].tolist())
            for i in range(num_clusters)
        ]
        continuation_counts = [
            group_by_continuation([
                compute_normalized_centroid(cnt, w, h)
                for cnt in clusters_info[clusters_info[:, 1] == i, 0]
            ])
            for i in range(num_clusters)
        ]
        figground_counts = [
            group_by_figure_ground(clusters_info[clusters_info[:, 1] == i, 0].tolist(), body_cnt, (h, w))
            for i in range(num_clusters)
        ]

        # Weighted average of grouping scores
        element_groups = {
            "similarity": statistics.fmean(similarity_counts) if similarity_counts else 0,
            "proximity": statistics.fmean(proximity_counts) if proximity_counts else 0,
            "closure": statistics.fmean(closure_counts) if closure_counts else 0,
            "continuation": statistics.fmean(continuation_counts) if continuation_counts else 0,
            "figure_ground": statistics.fmean(figground_counts) if figground_counts else 0,
        }

        weight = 1.0 / len(element_groups)
        gestalt_weights = {law: weight for law in element_groups}
        unity_score = calculate_unity_score(element_groups, len(el_contours), gestalt_weights)

    # --- Consolidate All Scores ---
    scores = {
        "balance_score": balance_score,
        "proportion_score": proportion_score,
        "symmetry_score": symmetry_score,
        "simplicity_score": simplicity_score,
        "harmony_score": harmony_score,
        "contrast_score": contrast_score,
        "unity_score": unity_score,
        "average_aesthetic_value": np.mean([
            balance_score, proportion_score, symmetry_score,
            simplicity_score, harmony_score, contrast_score, unity_score
        ])
    }

    return scores

def process_top(image_data):
    """
    End-to-end processing of an image to generate aesthetic scores.

    Args:
        image_data (str): Base64-encoded image string.

    Returns:
        dict: Aesthetic scores and annotated image base64.
    """
    # Decode image
    img = decode_image(image_data)

    # Run object detection
    yolo_detections = run_yolo_detection(img)
    roboflow_detections = multi_model_detect(img)

    # Combine all detections
    combined_detections = yolo_detections + roboflow_detections
    final_detections = resolve_label_conflicts(combined_detections)

    # Extract bboxes and labels
    bboxes = [det["bbox"] for det in final_detections]
    labels = [det["label"] for det in final_detections]

    # Run aesthetic scoring
    unified_score = process_image_with_bboxes(image_data, bboxes, labels)

    # Generate labeled result image
    labeled_image_b64 = encode_image_with_labels(img, bboxes, labels)
    unified_score["labeled_image"] = f"data:image/jpeg;base64,{labeled_image_b64}"

    return unified_score