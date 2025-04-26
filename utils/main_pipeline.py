import base64
import statistics
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from utils import *

def cluster_contours_by_kmeans(
    element_contours: list[np.ndarray],
    image_width: int,
    image_height: int,
    element_colors: list[tuple[float, float, float]]
) -> tuple[np.ndarray, int]:
    """
    Cluster a set of element contours by their normalized centroids using K-Means,
    retaining HSV color information per contour.

    Steps:
    1. Normalize centroid positions.
    2. Determine optimal number of clusters (k) using silhouette score.
    3. Assign cluster labels to each contour and store alongside color data.

    Args:
        element_contours (list of np.ndarray):
            OpenCV contours representing each detected element.
        image_width (int): Width of the original image (for normalization).
        image_height (int): Height of the original image (for normalization).
        element_colors (list of tuple[float, float, float]):
            HSV tuples per contour (H in [0,360), S/V in [0,255]).

    Returns:
        clusters_info (np.ndarray): Array of shape (N, 5) with:
            [contour, cluster_label, H, S, V] (dtype=object).
        num_clusters (int): Optimal number of clusters chosen via silhouette score.
    """
    n = len(element_contours)
    if n == 0:
        return np.empty((0, 5), dtype=object), 0
    if n == 1:
        # Single element: assign to one cluster
        single = np.array([[element_contours[0], 0, *element_colors[0]]], dtype=object)
        return single, 1

    # Step 1: compute normalized centroids for clustering
    centroids = np.array([
        compute_normalized_centroid(cnt, image_width, image_height)
        for cnt in element_contours
    ], dtype=np.float32)

    # Step 2: search for best k (up to 20 or n-1)
    best_score = -1.0
    best_labels = None
    max_k = min(20, n - 1)

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=1, max_iter=100)
        labels = km.fit_predict(centroids)
        score = silhouette_score(centroids, labels) if n > 2 else 1.0

        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    if best_labels is None:
        # Fallback: all belong to one cluster
        best_labels = np.zeros(n, dtype=int)
        best_k = 1

    # Step 3: assemble result array with [contour, label, H, S, V]
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
    Main scoring pipeline that computes all aesthetic principles for a given image and detections.

    Args:
        image_data (str): Base64-encoded image string.
        bboxes (List[List[float]]): List of bounding boxes in [x1, y1, x2, y2] format.
        labels (List[str]): Class labels corresponding to each bounding box.
        axis (str): Axis for symmetry ("vertical" or "horizontal").
        Nr (int): Reference element count for simplicity calculation.
        Dmin (float): Minimum simplicity degree.
        Dmax (float): Maximum simplicity degree.

    Returns:
        dict: Dictionary containing all aesthetic scores and final aesthetic value.
    """

    # --- Decode base64 image string into BGR OpenCV image ---
    img_str = image_data.split(',', 1)[1]
    data = base64.b64decode(img_str)
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image data"}

    h, w = img.shape[:2]

    # --- Identify the body as the largest box, rest are elements ---
    if bboxes:
        areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in bboxes]
        max_idx = int(np.argmax(areas))
        body_bbox = bboxes[max_idx]
        element_bboxes = [b for i, b in enumerate(bboxes) if i != max_idx]
    else:
        body_bbox = [0, 0, w, h]
        element_bboxes = []

    # --- Build contours from bounding boxes ---
    x1, y1, x2, y2 = map(int, body_bbox)
    body_cnt = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32).reshape(-1, 1, 2)
    body_size = cv2.contourArea(body_cnt)

    el_contours = []
    for (ex1, ey1, ex2, ey2) in element_bboxes:
        cnt = np.array([[ex1, ey1], [ex2, ey1], [ex2, ey2], [ex1, ey2]], dtype=np.int32).reshape(-1, 1, 2)
        el_contours.append(cnt)

    # --- Extract sizes, colors, centroids from contours ---
    info = extract_color_and_size_information(img, el_contours, body_cnt)
    el_colors    = info["element_colors"]
    el_sizes     = info["element_sizes"]
    body_color   = info["body_color"]
    el_centroids = info["element_centroids"]
    body_size    = info["body_size"]
    total_size   = body_size + sum(el_sizes)

    M = cv2.moments(body_cnt)
    if M['m00'] > 0:
        body_centroid = ((M['m10'] / M['m00']) / w, (M['m01'] / M['m00']) / h)
    else:
        body_centroid = (0.5, 0.5)

    # --- 1. Balance ---
    edges = edge_detect(img)
    s_balance = calculate_styling_balance(el_sizes, el_centroids, body_size, body_centroid, total_size)
    c_balance = calculate_color_balance(el_colors, body_color, el_centroids, el_sizes, body_size, body_centroid, total_size)
    t_balance = calculate_roughness_balance(el_contours, edges, el_centroids, el_sizes, body_cnt, body_centroid)
    balance_score = (s_balance + c_balance + t_balance) / 3

    # --- 2. Proportion ---
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

    # --- 3. Symmetry ---
    symmetry_score = calculate_symmetry_score(el_contours, img, axis)

    # --- 4. Simplicity ---
    try:
        edges = edge_detect(img)

        if el_contours and el_sizes:
            el_degrees = estimate_simplicity_from_roughness(el_contours, edges, el_sizes)
        else:
            print("[simplicity] WARNING: No element contours or sizes")
            el_degrees = [Dmin] * len(el_sizes)

        if body_cnt is not None and body_size > 0:
            body_degree = estimate_simplicity_from_roughness([body_cnt], edges, [body_size])[0]
        else:
            print("[simplicity] WARNING: No valid body contour, fallback to Dmin")
            body_degree = Dmin

        simplicity_score = calculate_simplicity_score(
            el_sizes, el_degrees, body_size, body_degree,
            reference_count=Nr, D_min=Dmin, D_max=Dmax,
            weights=(0.5, 0.5)
        )

    except Exception as e:
        print("[simplicity] ERROR:", e)
        import traceback
        traceback.print_exc()
        simplicity_score = 0.0

    # --- 5. Harmony & Contrast ---
    harmony_score  = calculate_harmony_score(el_colors, el_sizes, body_color, body_size)
    contrast_score = calculate_contrast_score(el_colors, el_sizes, body_color, body_size)

    # --- 6. Unity via clustering & Gestalt grouping ---
    clusters_info, num_clusters = cluster_contours_by_kmeans(el_contours, w, h, el_colors)

    if clusters_info.shape[0] == 0:
        print("[WARN] No element clusters found. Skipping unity computation.")
        return {
            "balance_score": balance_score,
            "proportion_score": proportion_score,
            "symmetry_score": symmetry_score,
            "simplicity_score": simplicity_score,
            "harmony_score": harmony_score,
            "contrast_score": contrast_score,
            "unity_score": 0.0,
            "average_aesthetic_value": np.mean([
                balance_score, proportion_score, symmetry_score,
                simplicity_score, harmony_score, contrast_score
            ])
        }

    # --- Compute Gestalt grouping counts per cluster ---
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
        group_by_figure_ground(
            clusters_info[clusters_info[:, 1] == i, 0].tolist(),
            body_cnt, (h, w)
        )
        for i in range(num_clusters)
    ]

    element_groups = {
        "similarity":     statistics.fmean(similarity_counts)    if similarity_counts else 0,
        "proximity":      statistics.fmean(proximity_counts)     if proximity_counts else 0,
        "closure":        statistics.fmean(closure_counts)       if closure_counts else 0,
        "continuation":   statistics.fmean(continuation_counts)  if continuation_counts else 0,
        "figure_ground":  statistics.fmean(figground_counts)     if figground_counts else 0
    }

    # Equal weights for all grouping laws
    w = 1.0 / len(element_groups) if element_groups else 1.0
    gestalt_weights = {law: w for law in element_groups}
    unity_score = calculate_unity_score(element_groups, len(el_contours), gestalt_weights)

    # --- Final scores ---
    scores = {
        "balance_score":     balance_score,
        "proportion_score":  proportion_score,
        "symmetry_score":    symmetry_score,
        "simplicity_score":  simplicity_score,
        "harmony_score":     harmony_score,
        "contrast_score":    contrast_score,
        "unity_score":       unity_score
    }

    scores["average_aesthetic_value"] = np.mean(list(scores.values()))
    return scores

def process_top(image_data):
    """
    Top-level processing function to compute aesthetic scores for a given input image.
    
    This function:
    1. Decodes a base64 image.
    2. Performs object detection using both YOLOv8 and multiple Roboflow models.
    3. Fuses and filters the detections using Non-Maximum Suppression and label conflict resolution.
    4. Extracts bounding boxes and class labels.
    5. Passes them to the aesthetic scoring engine.

    Args:
        image_data (str): A base64-encoded image URI string.

    Returns:
        dict: A dictionary of all computed aesthetic scores.
    """

    # --- Step 1: Decode base64 image into OpenCV BGR format ---
    img = decode_image(image_data)

    # --- Step 2: YOLOv8 detection ---
    yolo_detections = run_yolo_detection(img)

    # --- Step 3: Roboflow detection (already produces labeled output) ---
    roboflow_detections = multi_model_detect(img)
    for d in roboflow_detections:
        d["source"] = "roboflow"

    # Combine YOLO + Roboflow detections
    combined_detections = yolo_detections + roboflow_detections

    # --- Step 4: Resolve overlapping or conflicting boxes ---


    # Apply fusion-aware NMS and label resolution
    final_detections = resolve_label_conflicts(combined_detections)

    # Extract cleaned boxes and labels for scoring
    bboxes = [det["bbox"] for det in final_detections]
    labels = [det["label"] for det in final_detections]

    # --- Step 5: Run aesthetic scoring pipeline ---
    from utils.image_utils import encode_image_with_labels  # top of file if not imported already
    unified_score = process_image_with_bboxes(image_data, bboxes, labels)

    # Generate labeled output image
    labeled_image_b64 = encode_image_with_labels(img, bboxes, labels)

    # Return scores + image
    unified_score["labeled_image"] = f"data:image/jpeg;base64,{labeled_image_b64}"
    return unified_score
