import numpy as np
import cv2

from utils.image_utils import edge_detect
from utils.config import GESTALT_PROXIMITY_THRESHOLD, GESTALT_SIMILARITY_THRESHOLD

def compute_normalized_centroid(contour: np.ndarray, image_width: int, image_height: int) -> tuple[float, float]:
    """
    Compute the (x, y) centroid of an OpenCV contour, normalized to [0,1] by image dimensions.

    Args:
        contour (np.ndarray): Contour points (Nx1x2 array).
        image_width (int): Width of the source image in pixels.
        image_height (int): Height of the source image in pixels.

    Returns:
        tuple[float, float]: Normalized (x, y) centroid in [0,1] space.
    """
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = (M['m10'] / M['m00']) / image_width
        cy = (M['m01'] / M['m00']) / image_height
    else:
        # Fallback for degenerate contours
        cx, cy = 0.0, 0.0
    return cx, cy

def calculate_styling_balance(element_sizes, element_centroids, body_size, body_centroid, total_size):
    """
    Calculates the styling-based balance score.

    Measures how the weighted center of visual mass (elements + body)
    deviates horizontally from the geometric center of the layout (x = 0.5).

    Parameters:
    element_sizes : List[float]
    element_centroids : List[Tuple[float, float]]
    body_size : float
    body_centroid : Tuple[float, float]
    total_size : float

    Returns:
    float: balance score in [0,1], where 1.0 = perfect horizontal balance
    """
    if total_size <= 0 or body_size <= 0 or not element_sizes:
        return 0.1  # fail-safe return if no valid input

    # Compute weighted sum of element centroids
    weighted_sum_x = sum(cx * area for (cx, _), area in zip(element_centroids, element_sizes))

    # Include the body in the centroid calculation
    combined_centroid_x = (weighted_sum_x + body_centroid[0] * body_size) / total_size

    # Normalize deviation: shift x ∈ [0,1] → centered at 0.5 → [−0.5,0.5] → scale to [−1,1]
    deviation = (combined_centroid_x - 0.5) * 2  
    balance_score = 1 - abs(deviation)  # lower deviation = better balance

    return max(0.0, min(1.0, balance_score))  # clamp to [0, 1]


def calculate_color_balance(element_colors, body_color,
                            element_centroids, element_sizes,
                            body_size, body_centroid, total_size,
                            w_chr=0.7, w_v=0.3):
    """
    Calculates the color-based balance score.

    Uses spatial distribution of chroma (saturation) and inverse brightness (1-V)
    as a proxy for visual weight. The mass distribution is then compared against center.

    Parameters:
    element_colors : List[Tuple[float, float, float]]
    body_color : Tuple[float, float, float]
    element_centroids : List[Tuple[float, float]]
    element_sizes : List[float]
    body_size : float
    body_centroid : Tuple[float, float]
    total_size : float
    w_chr : float
    w_v : float

    Returns:
    float: combined color balance score ∈ [0,1]
    """
    # Extract chroma (S) and compute visual weight from value (V)
    chromas = np.array([col[1] for col in element_colors], dtype=float)
    vals = np.array([(10 - (col[2]/255*10)) for col in element_colors], dtype=float)

    # Compute chroma and brightness total mass (elements + body)
    mass_chr = chromas.dot(element_sizes) + body_color[1] * (body_size - sum(element_sizes))
    mass_val = vals.dot(element_sizes) + (10 - (body_color[2]/255*10)) * (body_size - sum(element_sizes))

    # Compute chroma-based balance
    if mass_chr <= 0:
        balance_chr = 1.0
    else:
        weighted_chr_x = (chromas * element_sizes * np.array([c for c, _ in element_centroids])).sum()
        weighted_chr_x += body_color[1] * body_centroid[0] * (body_size - sum(element_sizes))
        chr_centroid_x = weighted_chr_x / mass_chr
        balance_chr = 1 - abs((chr_centroid_x - 0.5) * 2)

    # Compute brightness-based balance
    if mass_val <= 0:
        balance_val = 1.0
    else:
        weighted_val_x = (vals * element_sizes * np.array([c for c, _ in element_centroids])).sum()
        weighted_val_x += (10 - (body_color[2]/255*10)) * body_centroid[0] * (body_size - sum(element_sizes))
        val_centroid_x = weighted_val_x / mass_val
        balance_val = 1 - abs((val_centroid_x - 0.5) * 2)

    # Clamp values and return weighted combination
    balance_chr = max(0.0, min(1.0, balance_chr))
    balance_val = max(0.0, min(1.0, balance_val))

    return w_chr * balance_chr + w_v * balance_val


def calculate_roughness_balance(element_contours, edges, element_centroids, element_sizes, body_contour, body_centroid):
    """
    Computes roughness balance using edge density (texture) as a proxy for visual weight.

    Implements Eq. (5) from Hu et al. (2022) to compute a weighted horizontal distribution
    of surface texture using edge map masks.

    Parameters:
    element_contours : List[np.ndarray]
    edges : np.ndarray
    element_centroids : List[Tuple[float,float]]
    element_sizes : List[float]
    body_contour : np.ndarray
    body_centroid : Tuple[float,float]

    Returns:
    float: roughness balance score ∈ [0, 1]
    """
    body_area = cv2.contourArea(body_contour)
    if body_area <= 0:
        return 0.1

    # Roughness of each element = # edge pixels inside / area
    rou_i = []
    sum_Cix_Si = 0.0

    for cnt, (cx, _), Si in zip(element_contours, element_centroids, element_sizes):
        # Create binary mask for each element
        mask_el = np.zeros(edges.shape, dtype=np.uint8)
        cv2.drawContours(mask_el, [cnt], -1, 255, thickness=cv2.FILLED)

        edge_count = int(np.count_nonzero(edges & mask_el))
        rou = edge_count / (Si + 1e-5)
        rou_i.append(rou)
        sum_Cix_Si += cx * Si

    # Roughness of the body region
    mask_b = np.zeros(edges.shape, dtype=np.uint8)
    cv2.drawContours(mask_b, [body_contour], -1, 255, thickness=cv2.FILLED)
    body_edge_count = int(np.count_nonzero(edges & mask_b))
    rou_b = body_edge_count / (body_area + 1e-5)

    # Numerator and denominator as per Eq. (5)
    weighted_sum = sum(r * cx * s for r, (cx, _), s in zip(rou_i, element_centroids, element_sizes))
    numerator = weighted_sum + rou_b * (body_centroid[0] * body_area - sum_Cix_Si)
    denom = sum(r * s for r, s in zip(rou_i, element_sizes)) + rou_b * (body_area - sum(element_sizes))

    if denom == 0:
        return 1.0

    # Normalization factor: how far overall centroid deviates from x=0.5
    combined_cx = (
        (sum(cx * s for (cx, _), s in zip(element_centroids, element_sizes)) + body_centroid[0] * body_area)
        / (body_area + sum(element_sizes))
    )
    norm_factor = (combined_cx - 0.5) * 2

    balance = 1 - abs((numerator / denom) * norm_factor)
    return max(0.0, min(1.0, balance))

def calculate_proportion_score(elements: list[dict], body: dict, Rc: float = 1.618, 
                               weights: tuple[float, float, float] = (1/3, 1/3, 1/3)) -> float:
    """
    Calculates the overall proportion score (P) for a visual composition, based on:

        1. PWL – Width-to-Length Ratio Conformity
        2. PL  – Positional Ratio Conformity
        3. PS  – Scale Ratio Conformity

    Each sub-score evaluates how closely the composition matches the classical
    reference ratio Rc (e.g., the Golden Ratio = 1.618). Weighted sum is used.

    Parameters:
    elements : list of dict
        Detected objects, each with:
            - 'width': pixel width of the bounding box
            - 'height': pixel height of the bounding box
            - 'centroid_x': x-center ∈ [0,1]
            - 'centroid_y': y-center ∈ [0,1]
            - 'size': area in pixels
    body : dict
        Same keys as above; represents the entire body layout.
    Rc : float
        Reference classical ratio (default = 1.618).
    weights : tuple of 3 floats
        Weights for (PWL, PL, PS), must sum to 1.

    Returns:
    float
        Overall proportion score ∈ [0,1]
    """
    w_pwl, w_pl, w_ps = weights
    if not np.isclose(w_pwl + w_pl + w_ps, 1.0):
        raise ValueError("Weights must sum to 1.")

    # Extract body parameters
    Bw, Bh, Sb = body["width"], body["height"], body["size"]
    total_area = Sb + sum(el["size"] for el in elements)
    element_area = total_area - Sb

    # --- 1. Width-to-Length Ratio (PWL) ---
    # Measures how element and body aspect ratios align with Rc
    def ratio(a, b): return (a / b) if a <= b else (b / a)

    PWL_diff = sum(abs(ratio(el["width"], el["height"]) - Rc) * el["size"] for el in elements)
    PWL_diff += abs(ratio(Bw, Bh) - Rc) * Sb
    PWL = max(0.0, 1.0 - (PWL_diff / (total_area * Rc)))  # scaled deviation

    # --- 2. Position Ratio (PL) ---
    # Measures how close element centroids are to reference Rc in normalized [0,1]
    PLx = 1.0 - (1.0 / Rc) * sum(abs(el["centroid_x"] - Rc) * (el["size"] / element_area) for el in elements)
    PLy = 1.0 - (1.0 / Rc) * sum(abs(el["centroid_y"] - Rc) * (el["size"] / element_area) for el in elements)
    PL = max(0.0, min(1.0, 0.5 * (PLx + PLy)))  # average and clamp to [0,1]

    # --- 3. Scale Ratio (PS) ---
    # Compares element size ratios (width/body_width, height/body_height) to Rc
    PSx = 1.0 - (1.0 / Rc) * sum(abs((el["width"] / Bw) - Rc) * (el["size"] / element_area) for el in elements)
    PSy = 1.0 - (1.0 / Rc) * sum(abs((el["height"] / Bh) - Rc) * (el["size"] / element_area) for el in elements)
    PS = max(0.0, min(1.0, 0.5 * (PSx + PSy)))

    # --- Combine with weights ---
    return w_pwl * PWL + w_pl * PL + w_ps * PS

def calculate_symmetry_score(element_contours: list[np.ndarray],
                             image: np.ndarray,
                             axis: str = "vertical") -> float:
    """
    Calculates a symmetry score for a multi-object layout by analyzing the distribution
    of edge pixels within the detected object regions across a chosen axis.

    Parameters:
    element_contours : list of np.ndarray
        Contours of all detected visual elements in the layout.
    image : np.ndarray
        Original image in BGR format.
    axis : str
        Symmetry axis — "vertical" (default) for left/right, or "horizontal" for top/bottom.

    Returns:
    float
        Symmetry score ∈ [0, 1], where 1.0 means perfectly mirrored edge content.
    """
    # Step 1: Edge detection (uses adaptive Canny + morphology cleanup)
    edges = edge_detect(image)  # must be defined in your codebase

    # Step 2: Create binary mask covering only element regions
    element_mask = np.zeros_like(edges)
    cv2.drawContours(element_mask, element_contours, -1, color=255, thickness=cv2.FILLED)

    # Step 3: Extract only the edges within those regions
    masked_edges = cv2.bitwise_and(edges, edges, mask=element_mask)

    h, w = masked_edges.shape

    # Step 4: Count edge pixels on both sides of the selected axis
    if axis.lower() == "vertical":
        # Left-right symmetry
        left_half  = masked_edges[:, :w // 2]
        right_half = masked_edges[:, w // 2:]
        right_flipped = cv2.flip(right_half, 1)  # flip horizontally
        count_left  = int(np.count_nonzero(left_half))
        count_right = int(np.count_nonzero(right_flipped))
    else:
        # Top-bottom symmetry
        top_half     = masked_edges[:h // 2, :]
        bottom_half  = masked_edges[h // 2:, :]
        bottom_flip  = cv2.flip(bottom_half, 0)  # flip vertically
        count_left   = int(np.count_nonzero(top_half))
        count_right  = int(np.count_nonzero(bottom_flip))

    # Step 5: Symmetry score = 1 - normalized difference
    diff   = abs(count_left - count_right)
    larger = max(count_left, count_right)
    return 1.0 - (diff / larger) if larger > 0 else 1.0

def calculate_simplicity_score(element_sizes: list[float],
                               element_degrees: list[float],
                               body_size: float,
                               body_degree: float,
                               reference_count: int = 0,
                               D_max: float = 10.0,
                               D_min: float = 1.0,
                               weights: tuple[float, float] = (0.5, 0.5)) -> float:
    """
    Compute a simplicity score for a multi‑object layout by combining:
      1. Simplicity_N — reduction in number of elements vs. a reference design.
      2. Simplicity_D — normalized visual simplicity of individual elements and the body.

    Parameters:
    element_sizes : list[float]
        Area in pixels² of each element.
    element_degrees : list[float]
        Simplicity degree Dᵢ for each element ∈ [D_min, D_max].
    body_size : float
        Area of the overall body contour.
    body_degree : float
        Simplicity degree of the body ∈ [D_min, D_max].
    reference_count : int
        Reference number of elements in a "simplest" design. Default: 0.
    D_max : float
        Maximum possible complexity score (default: 10.0).
    D_min : float
        Minimum possible complexity score (default: 1.0).
    weights : tuple[float, float]
        Weights (wN, wD) for the count vs. degree components. Must sum to 1.

    Returns:
    float
        Simplicity score ∈ [0, 1]. Higher = simpler layout.
    """
    wN, wD = weights
    if not np.isclose(wN + wD, 1.0):
        raise ValueError("Weights must sum to 1.0")

    # Step 1: Simplicity from element count
    N_actual = len(element_sizes)
    simplicity_N = 1.0 - (reference_count / N_actual) if N_actual > 0 else 0.0

    # Step 2: Simplicity from visual degree (weighted by area)
    denom = D_max - D_min
    total_area = sum(element_sizes) + body_size
    normalized_sum = 0.0

    for size, D_i in zip(element_sizes, element_degrees):
        norm = (D_max - D_i) / denom  # high D_i = complex = low norm
        normalized_sum += norm * size

    # Add body component
    norm_body = (D_max - body_degree) / denom
    normalized_sum += norm_body * body_size

    simplicity_D = normalized_sum / total_area if total_area > 0 else 0.0

    # Combine both components
    return wN * simplicity_N + wD * simplicity_D

def estimate_simplicity_from_roughness(contours, edge_map, sizes):
    """
    Estimate a list of visual complexity degrees (Dᵢ) based on edge density,
    which serves as a proxy for texture/visual simplicity.

    Higher edge density → higher Dᵢ (more visually complex).

    Parameters:
    contours : list[np.ndarray]
        Contours of each visual element.
    edge_map : np.ndarray
        Binary edge map (same size as image).
    sizes : list[float]
        Areas of each contour (in pixels).

    Returns:
    list[float]
        Degree Dᵢ for each element, ∈ [D_min, D_max].
    """
    D_min, D_max = 1.0, 10.0
    degrees = []

    for cnt, area in zip(contours, sizes):
        # Create binary mask for contour
        mask = np.zeros_like(edge_map)
        cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)

        # Count edge pixels within this contour
        edge_count = np.count_nonzero(cv2.bitwise_and(edge_map, edge_map, mask=mask))
        roughness = edge_count / (area + 1e-5)

        # Linearly map roughness to [D_min, D_max]
        d_i = np.clip(D_min + (D_max - D_min) * roughness, D_min, D_max)
        degrees.append(d_i)

    return degrees

def calculate_harmony_score(element_colors: list[tuple[float, float, float]],
                            element_sizes: list[float],
                            body_color: tuple[float, float, float],
                            body_size: float) -> float:
    """
    Calculates a color harmony score based on the distribution of hues across 
    detected interior elements.

    This implementation blends:
      - Complementary harmony: how evenly hues are distributed around the color wheel.
      - Analogous harmony: how closely grouped the hues are.

    Parameters:
    element_colors : list of (H, S, V)
        Hue (0–360), Saturation, Value for each element.
    element_sizes : list of float
        Area of each corresponding element (used as weight).
    body_color : (H, S, V)
        Not used directly, included for interface consistency.
    body_size : float
        Not used directly, included for interface consistency.

    Returns:
    float
        Harmony score ∈ [0, 1]. Higher = more visually harmonious layout.
    """
    try:
        # fallback case if data is missing or invalid
        if not element_colors or not element_sizes or body_size <= 0:
            return 1.0

        # ensure color and size lists match
        if len(element_colors) != len(element_sizes):
            raise ValueError(f"Mismatch: {len(element_colors)} colors vs {len(element_sizes)} sizes")

        # extract hues and normalize weights
        hues = np.array([h for h, _, _ in element_colors])
        weights = np.array(element_sizes)
        weight_sum = weights.sum()
        if weight_sum == 0:
            return 1.0
        weights /= weight_sum

        # === Complementary Harmony ===
        # Measures circular dispersion on hue wheel using sine/cosine projection
        radians = 2 * np.pi * hues / 360
        sin_sum = np.sum(weights * np.sin(radians))
        cos_sum = np.sum(weights * np.cos(radians))
        complementary = 1.0 - (abs(sin_sum) + abs(cos_sum)) / 2

        # === Analogous Harmony ===
        # Measures closeness to weighted mean hue
        mean_hue = np.average(hues, weights=weights)
        hue_diffs = np.abs(hues - mean_hue)
        hue_diffs = np.minimum(hue_diffs, 360 - hue_diffs)  # wraparound fix
        avg_diff = np.average(hue_diffs, weights=weights)
        analogous = 1.0 - (avg_diff / 180)  # normalize to [0,1]

        # Combined score as mean of the two harmonies
        harmony = (complementary + analogous) / 2
        return max(0.0, min(harmony, 1.0))  # final clamp

    except Exception as e:
        print(f"[ERROR @ calculate_harmony_score]: {e}")
        return 1.0

def calculate_contrast_score(element_colors: list[tuple[float, float, float]],
                             element_sizes: list[float],
                             body_color: tuple[float, float, float],
                             body_size: float) -> float:
    """
    Compute a multi-object color contrast score based on the visual distance
    between each element’s HSV value and the body (background) region.

    Parameters:
    element_colors : list of (H, S, V)
        Hue (0–360), Saturation, and Value for each element.
    element_sizes : list of float
        Pixel area of each corresponding element contour.
    body_color : (H, S, V)
        HSV tuple of the full-body/background region.
    body_size : float
        Area (in pixels) of the background contour.

    Returns:
    float
        Contrast score in [0, 1]. A higher score indicates stronger
        perceptual separation between elements and the background.
    """
    # fallback: if body invalid or no elements, assume full contrast
    if not element_colors or body_size <= 0:
        return 1.0

    # max possible HSV distance (used for normalization)
    max_dist = np.sqrt(360**2 + 255**2 + 255**2)
    Hb, Sb, Vb = body_color

    weighted_sum = 0.0
    for (He, Se, Ve), area in zip(element_colors, element_sizes):
        # Euclidean distance between element and body in HSV space
        dist = np.sqrt((He - Hb)**2 + (Se - Sb)**2 + (Ve - Vb)**2)
        normalized = dist / max_dist
        # weight by how large the element is relative to body
        weighted_sum += normalized * (area / body_size)

    # final contrast = inverse of similarity
    contrast = 1.0 - weighted_sum
    return max(0.0, min(contrast, 1.0))

def calculate_unity_score(
    element_groups: dict[str, int],
    total_elements: int,
    weights: dict[str, float]) -> float:
    """
    Calculate an overall unity score by combining the number of perceptual groups 
    found under multiple Gestalt principles like similarity, proximity, etc.

    For each grouping law j:
        N_j = number of groups under that law
        U_j = 1 - (N_j - 1) / total_elements

    The final score is a weighted sum over all laws:
        unity = ∑ w_j * U_j, where ∑ w_j = 1.

    Parameters:
    element_groups : dict[str, int]
        A mapping from each Gestalt principle to its number of visual groups.
        Example: {"proximity": 3, "similarity": 2}
    total_elements : int
        Total number of visual elements considered in the layout.
    weights : dict[str, float]
        A dictionary mapping each Gestalt principle to its corresponding weight.
        Must sum to 1.0.

    Returns:
    float
        A unity score in [0, 1]. Higher values indicate stronger perceptual unity.
    """
    # Trivial case: 1 or 0 elements are always unified
    if total_elements <= 1:
        return 1.0

    # Sanity check: sum of weights must be ~1.0
    total_w = sum(weights.get(law, 0) for law in element_groups)
    if not np.isclose(total_w, 1.0):
        raise ValueError(f"Gestalt weights must sum to 1.0 (got {total_w:.3f})")

    unity = 0.0
    for law, N_j in element_groups.items():
        # Unity per law: 1 means perfectly unified under this principle
        U_j = 1.0 - float(N_j - 1) / total_elements
        unity += weights[law] * U_j  # weighted sum

    return min(max(unity, 0.0), 1.0)  # clamp to [0, 1]

def group_by_similarity(
    element_colors: list[tuple[float, float, float]],
    threshold_hsv: float = GESTALT_SIMILARITY_THRESHOLD) -> int:
    """
    Groups interior elements by color similarity using HSV distance in 3D space.

    The function converts OpenCV HSV tuples into interpretable hue (degrees) and 
    saturation/value (percentage), then builds a connectivity graph where two elements 
    are considered “similar” if their Euclidean distance is under `threshold_hsv`.
    The number of connected components in this graph is the number of color groups.

    Args:
        element_colors (list of tuple):
            Per-element mean HSV values as (H ∈ [0,180], S ∈ [0,255], V ∈ [0,255]).
        threshold_hsv (float):
            Maximum 3D HSV distance for two elements to be considered similar.

    Returns:
        int:
            Number of similarity-based groups (lower means more unified colors).
    """
    # 1. Convert OpenCV HSV → (H in degrees, S/V in %)
    conv = [
        (h * 2.0, (s / 255.0) * 100.0, (v / 255.0) * 100.0)
        for h, s, v in element_colors
    ]
    n = len(conv)
    if n == 0:
        return 0  # no elements, no groups

    # 2. Build adjacency matrix (undirected)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            dh = conv[i][0] - conv[j][0]
            ds = conv[i][1] - conv[j][1]
            dv = conv[i][2] - conv[j][2]
            dist = (dh*dh + ds*ds + dv*dv) ** 0.5
            if dist < threshold_hsv:
                adj[i, j] = adj[j, i] = True

    # 3. Count connected components using DFS
    visited = [False] * n

    def dfs(u: int):
        stack = [u]
        visited[u] = True
        while stack:
            cur = stack.pop()
            for v in range(n):
                if adj[cur, v] and not visited[v]:
                    visited[v] = True
                    stack.append(v)

    groups = 0
    for idx in range(n):
        if not visited[idx]:
            dfs(idx)
            groups += 1

    return groups

def group_by_proximity(
    element_centroids: list[tuple[float, float]],
    threshold: float = GESTALT_PROXIMITY_THRESHOLD) -> int:
    """
    Groups elements by spatial proximity in normalized coordinate space [0,1].

    Two elements are “proximate” if the Euclidean distance between their centroids
    is less than `threshold`. We form an undirected graph where such elements are 
    connected, and count the number of connected components as spatial groups.

    Args:
        element_centroids (list of tuple):
            Normalized (x, y) coordinates of each element ∈ [0,1].
        threshold (float):
            Maximum distance for two elements to be grouped.

    Returns:
        int:
            Number of proximity-based clusters (lower = more packed).
    """
    n = len(element_centroids)
    if n == 0:
        return 0  # no elements, no clusters

    # Build adjacency matrix for proximity
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        xi, yi = element_centroids[i]
        for j in range(i + 1, n):
            xj, yj = element_centroids[j]
            dist = np.hypot(xi - xj, yi - yj)
            if dist < threshold:
                adj[i, j] = adj[j, i] = True

    # Count connected components using DFS
    visited = [False] * n

    def dfs(u: int):
        stack = [u]
        visited[u] = True
        while stack:
            cur = stack.pop()
            for v in range(n):
                if adj[cur, v] and not visited[v]:
                    visited[v] = True
                    stack.append(v)

    groups = 0
    for idx in range(n):
        if not visited[idx]:
            dfs(idx)
            groups += 1

    return groups

def group_by_closure(
    element_contours: list[np.ndarray],
    overlap_threshold: float = 0.3) -> int:
    """
    Groups interior elements by the Gestalt law of closure using bounding box overlap.

    Two elements are linked (grouped) if the area of intersection between their bounding boxes 
    is at least `overlap_threshold` × the smaller box area.

    Args:
        element_contours (list of np.ndarray):
            Contours for each segmented interior element.
        overlap_threshold (float):
            Minimum overlap ratio (0–1) required to treat two elements as one group.

    Returns:
        int:
            Number of closure-based groups (connected components).
    """
    rects = [cv2.boundingRect(cnt) for cnt in element_contours]
    n = len(rects)
    if n == 0:
        return 0

    # Initialize adjacency matrix
    adj = np.zeros((n, n), dtype=bool)

    for i in range(n):
        x1, y1, w1, h1 = rects[i]
        area1 = w1 * h1
        for j in range(i + 1, n):
            x2, y2, w2, h2 = rects[j]
            area2 = w2 * h2

            # Calculate intersection area
            inter_w = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            inter_h = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            intersection = inter_w * inter_h

            min_area = min(area1, area2)
            if min_area > 0 and (intersection / min_area) >= overlap_threshold:
                adj[i, j] = adj[j, i] = True

    # DFS to count connected components
    visited = [False] * n

    def dfs(u: int):
        stack = [u]
        visited[u] = True
        while stack:
            cur = stack.pop()
            for v in range(n):
                if adj[cur, v] and not visited[v]:
                    visited[v] = True
                    stack.append(v)

    groups = 0
    for idx in range(n):
        if not visited[idx]:
            dfs(idx)
            groups += 1

    return groups

def group_by_continuation(
    element_centroids: list[tuple[float, float]],
    angle_threshold_degrees: float = 8.0,
    dist_ratio_tolerance: float = 0.3) -> int:
    """
    Groups elements by Gestalt principle of continuation:
    If elements align along a nearly straight path, they are perceived as continuous.

    This function checks all triplets of elements and connects outer pairs (i,k) 
    if their angle at midpoint (j) is small and the distances j-i ≈ j-k.

    Args:
        element_centroids (list of tuple[float, float]):
            Normalized centroid coordinates of each element.
        angle_threshold_degrees (float):
            Maximum angle at midpoint (in degrees) to consider a straight path.
        dist_ratio_tolerance (float):
            Allowed deviation in distance ratio for alignment (0.3 = ±30%).

    Returns:
        int:
            Number of continuation-based groups.
    """
    n = len(element_centroids)
    if n < 3:
        return n if n > 1 else 1

    adj = np.zeros((n, n), dtype=bool)

    def _angle_and_dists(a, b, c):
        """Computes angle at point b formed by a–b–c and distances."""
        ab = (a[0] - b[0], a[1] - b[1])
        cb = (c[0] - b[0], c[1] - b[1])
        dot = ab[0]*cb[0] + ab[1]*cb[1]
        mag_ab = np.hypot(*ab)
        mag_cb = np.hypot(*cb)
        if mag_ab == 0 or mag_cb == 0:
            return 180.0, mag_ab, mag_cb  # degenerate angle
        cosθ = np.clip(dot / (mag_ab * mag_cb), -1.0, 1.0)
        angle = np.degrees(np.arccos(cosθ))
        return angle, mag_ab, mag_cb

    # Link i–k if ∃ j s.t. angle(i–j–k) is small and distances are similar
    for i in range(n):
        for k in range(i+1, n):
            for j in range(n):
                if j in (i, k):
                    continue
                ang, d_ji, d_jk = _angle_and_dists(
                    element_centroids[i],
                    element_centroids[j],
                    element_centroids[k]
                )
                if ang < angle_threshold_degrees:
                    if d_jk > 0:
                        ratio = d_ji / d_jk
                        if 1 - dist_ratio_tolerance <= ratio <= 1 + dist_ratio_tolerance:
                            adj[i, k] = adj[k, i] = True
                            break  # no need to test other j

    # DFS to find connected components
    visited = [False] * n

    def _dfs(start):
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            for v in range(n):
                if adj[u, v] and not visited[v]:
                    visited[v] = True
                    stack.append(v)

    groups = 0
    for idx in range(n):
        if not visited[idx]:
            _dfs(idx)
            groups += 1

    return groups

def group_by_figure_ground(
    element_contours: list[np.ndarray],
    body_contour: np.ndarray,
    image_shape: tuple[int, int],
    area_ratio_threshold: float = 0.5) -> int:
    """
    Groups elements by figure–ground: elements whose filled area overlaps the
    body by >= area_ratio_threshold are 'figure', others 'ground'.

    We then count how many distinct figure/ground categories exist:
      - All figure or all ground => 1 group
      - Mixed => 2 groups

    Args:
        element_contours: list of small contours (np.int32 Nx1x2).
        body_contour: main body contour (np.int32 Mx1x2).
        image_shape: (height, width) of the canvas to rasterize masks.
        area_ratio_threshold: fraction of element area that must lie inside body
                              to be considered "inside" (default 0.5).

    Returns:
        int: 0 if no elements; otherwise 1 or 2.
    """
    h, w = image_shape
    n = len(element_contours)
    if n == 0:
        return 0

    # rasterize body once
    body_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(body_mask, [body_contour], -1, 255, thickness=cv2.FILLED)

    inside_flags = []
    for cnt in element_contours:
        # rasterize element
        el_mask = np.zeros_like(body_mask)
        cv2.drawContours(el_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # compute areas
        area_el = cv2.countNonZero(el_mask)
        if area_el == 0:
            inside_flags.append(False)
            continue

        intersection = cv2.bitwise_and(el_mask, body_mask)
        area_int = cv2.countNonZero(intersection)

        # figure if majority of element is inside
        inside_flags.append((area_int / area_el) >= area_ratio_threshold)

    inside_count = sum(inside_flags)
    # all outside or all inside → 1 grouping; mixed → 2
    return 1 if (inside_count == 0 or inside_count == n) else 2

