"""CPU document detection and perspective correction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

from image_toolkit.processing import MAX_PIXELS


@dataclass
class Detection:
    corners: list[list[float]] | None
    confidence: float
    method: str
    message: str


def order_points(points):
    """Return four points in TL, TR, BR, BL order, rejecting ambiguous geometry."""
    pts = np.asarray(points, dtype="float32").reshape(4, 2)
    if not np.isfinite(pts).all() or len(np.unique(np.round(pts, 3), axis=0)) != 4:
        raise ValueError("Select four distinct, finite corners.")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    polygon = pts[np.argsort(angles)]
    # Rotate the clockwise polygon so its top-left (minimum x+y) is first.
    start = int(np.argmin(polygon.sum(axis=1)))
    polygon = np.roll(polygon, -start, axis=0)
    first, second = polygon[1] - polygon[0], polygon[2] - polygon[1]
    if first[0] * second[1] - first[1] * second[0] < 0:
        polygon = polygon[[0, 3, 2, 1]]
    if not cv_is_convex(polygon):
        raise ValueError("Corners must form a convex quadrilateral in TL, TR, BR, BL order.")
    return polygon.astype("float32")


def cv_is_convex(points):
    edges = np.roll(points, -1, axis=0) - points
    next_edges = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    return bool(np.all(turns > 1e-3) or np.all(turns < -1e-3))


def validate_corners(points, image_size):
    """Validate the user's ordered TL, TR, BR, BL points without reordering them."""
    pts = np.asarray(points, dtype="float32").reshape(4, 2)
    width, height = image_size
    if not np.isfinite(pts).all() or (pts[:, 0] < 0).any() or (pts[:, 0] >= width).any() or (pts[:, 1] < 0).any() or (pts[:, 1] >= height).any():
        raise ValueError("All four corners must be inside the image.")
    if len(np.unique(np.round(pts, 2), axis=0)) != 4:
        raise ValueError("Each corner must be at a different location.")
    if not cv_is_convex(pts):
        raise ValueError("Corners cross or form a concave shape. Arrange them clockwise from top-left.")
    area = abs(float(np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], -1)))) / 2
    if area < width * height * .003:
        raise ValueError("The selected corner area is too small.")
    return pts


def preview_to_source(point, preview_size, source_size):
    preview_width, preview_height = preview_size
    source_width, source_height = source_size
    if min(preview_width, preview_height, source_width, source_height) < 1:
        raise ValueError("Preview and source dimensions must be positive.")
    x = min(source_width - 1, max(0, round(point[0] * source_width / preview_width)))
    y = min(source_height - 1, max(0, round(point[1] * source_height / preview_height)))
    return [float(x), float(y)]


def perspective_transform(image: Image.Image, points) -> Image.Image:
    ordered = validate_corners(points, image.size)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Document scanning requires opencv-python-headless. Install the base project requirements.") from exc
    width = int(round(max(np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3])))) + 1
    height = int(round(max(np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1])))) + 1
    if width < 2 or height < 2 or width * height > MAX_PIXELS:
        raise ValueError("The selected corners do not form a valid document area.")
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    rgb = np.asarray(image.convert("RGB"))
    matrix = cv2.getPerspectiveTransform(ordered, target)
    corrected = cv2.warpPerspective(rgb, matrix, (width, height), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(corrected)


def _quad_quality(points, rgb, edge_maps):
    import cv2

    height, width = rgb.shape[:2]
    try:
        quad = order_points(points)
    except (ValueError, TypeError):
        return -1.0
    if (quad[:, 0].min() < -width * .02 or quad[:, 0].max() > width * 1.02
            or quad[:, 1].min() < -height * .02 or quad[:, 1].max() > height * 1.02):
        return -1.0
    area = abs(float(np.dot(quad[:, 0], np.roll(quad[:, 1], -1)) - np.dot(quad[:, 1], np.roll(quad[:, 0], -1)))) / 2
    ratio = area / (width * height)
    if ratio < .025 or ratio > .94:
        return -1.0
    lengths = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    if min(lengths) < min(width, height) * .045:
        return -1.0
    # A broad range permits receipts, landscape forms and certificates.
    horizontal = (lengths[0] + lengths[2]) / 2
    vertical = (lengths[1] + lengths[3]) / 2
    aspect = max(horizontal, vertical) / max(1.0, min(horizontal, vertical))
    aspect_score = 1.0 if aspect <= 7 else max(0.0, 1.0 - (aspect - 7) / 8)
    cosines = []
    for i in range(4):
        a, b = quad[i - 1] - quad[i], quad[(i + 1) % 4] - quad[i]
        cosines.append(abs(float(np.dot(a, b) / max(1.0, np.linalg.norm(a) * np.linalg.norm(b)))))
    right_angle_score = max(0.0, 1.0 - float(np.mean(cosines)) * 1.5)
    # Edge support is sampled along each side with a small tolerance.
    edge_support = 0.0
    for edge_map in edge_maps:
        support = []
        for i in range(4):
            a, b = quad[i], quad[(i + 1) % 4]
            count = max(20, int(np.linalg.norm(b - a) / 4))
            xs = np.clip(np.linspace(a[0], b[0], count).astype(int), 0, width - 1)
            ys = np.clip(np.linspace(a[1], b[1], count).astype(int), 0, height - 1)
            hits = []
            for x, y in zip(xs, ys):
                patch = edge_map[max(0, y-2):min(height, y+3), max(0, x-2):min(width, x+3)]
                hits.append(bool(np.any(patch)))
            support.append(float(np.mean(hits)))
        edge_support = max(edge_support, float(np.mean(support)))
    # Give page-like interiors some weight so a generic high-contrast frame is not
    # treated as certain solely because its outline is rectangular.
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype("int32"), 255)
    mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=2)
    interior = rgb[mask > 0]
    if len(interior):
        values = interior.astype("float32")
        gray_values = values @ np.array([.299, .587, .114], dtype="float32")
        saturation = (values.max(axis=1) - values.min(axis=1)) / np.maximum(values.max(axis=1), 1)
        brightness_evidence = min(1.0, max(0.0, (float(np.percentile(gray_values, 70)) - 70) / 130))
        paper_color_evidence = 1.0 - min(1.0, float(np.median(saturation)) / .55)
        interior_edge_density = float(np.count_nonzero(edge_maps[0][mask > 0])) / max(1, len(interior))
        detail_evidence = min(1.0, interior_edge_density / .035)
        page_evidence = .45 * brightness_evidence + .35 * paper_color_evidence + .20 * detail_evidence
    else:
        page_evidence = 0.0
    area_score = min(1.0, ratio * 2.2)
    return (.25 * area_score + .18 * aspect_score + .16 * right_angle_score
            + .24 * min(1.0, edge_support * 7) + .17 * page_evidence)


def detect_document(image: Image.Image) -> Detection:
    """Find document candidates from contrast, threshold, connected-region and edge maps.

    This deliberately reports uncertain candidates rather than silently cropping. It uses
    classical CPU vision; a visually indistinguishable page/background cannot be inferred.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Document scanning requires opencv-python-headless.") from exc
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    rgb = np.asarray(oriented)
    scale = min(1.0, 1400 / max(rgb.shape[:2]))
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges_a = cv2.Canny(enhanced, 25, 90)
    edges_b = cv2.Canny(enhanced, 55, 160)
    adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    adaptive_inv = cv2.bitwise_not(adaptive)
    edge_close = cv2.morphologyEx(cv2.bitwise_or(edges_a, edges_b), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    maps = [edges_a, edges_b, edge_close]
    candidates = []
    method_by_map = [(edges_a, "contrast edges"), (edges_b, "strong edges"), (edge_close, "joined edges"),
                     (adaptive, "adaptive regions"), (adaptive_inv, "adaptive regions")]
    for binary, method in method_by_map:
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:35]:
            area = cv2.contourArea(contour)
            if area < h * w * .025:
                continue
            perimeter = cv2.arcLength(contour, True)
            for epsilon in (.012, .02, .03, .045):
                approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    candidates.append((approx.reshape(4, 2), method))
                    break
            # A region can have ragged edges; its rotated rectangle is only a candidate,
            # and receives a low score unless side evidence supports it.
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            if cv2.contourArea(contour) / max(1.0, rect[1][0] * rect[1][1]) > .50:
                candidates.append((box, method + " rectangle"))
    # Hough line segments offer a second source when page borders are broken into pieces.
    lines = cv2.HoughLinesP(edge_close, 1, np.pi / 180, threshold=max(30, min(h, w) // 12),
                            minLineLength=max(25, min(h, w) // 9), maxLineGap=max(12, min(h, w) // 35))
    if lines is not None:
        segments = []
        for item in lines[:, 0, :]:
            x1, y1, x2, y2 = map(float, item)
            length = np.hypot(x2-x1, y2-y1)
            theta = np.arctan2(y2-y1, x2-x1) % np.pi
            if length > min(h, w) * .12:
                segments.append((theta, length, (x1, y1, x2, y2)))
        # Pair long near-parallel line segments from two approximately perpendicular families.
        pairs = list(combinations(sorted(segments, key=lambda row: row[1], reverse=True)[:18], 2))
        line_families = []
        for first, second in pairs:
            angle_delta = abs(first[0]-second[0]); angle_delta = min(angle_delta, np.pi-angle_delta)
            if angle_delta < np.deg2rad(18):
                line_families.append((first, second))
        for (a, b), (c, d) in combinations(line_families[:24], 2):
            family_angle = abs(a[0]-c[0]); family_angle = min(family_angle, np.pi-family_angle)
            if abs(family_angle - np.pi/2) > np.deg2rad(28):
                continue
            intersections = [_line_intersection(l1[2], l2[2]) for l1, l2 in ((a,c),(a,d),(b,d),(b,c))]
            if all(p is not None for p in intersections):
                candidates.append((np.array(intersections, dtype="float32"), "Hough line groups"))
    scored = []
    for points, method in candidates:
        score = _quad_quality(points, small, maps)
        if score > 0:
            scored.append((score, points, method))
    # Last, local CPU fallback: GrabCut seeded with a probable-foreground interior.
    # It can recover a page separated by color/shadow rather than a closed outline.
    if not scored or max(row[0] for row in scored) < .62:
        try:
            grab_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
            margin = max(4, int(min(h, w) * .025))
            grab_mask[margin:h-margin, margin:w-margin] = cv2.GC_PR_FGD
            grab_mask[int(h*.45):max(int(h*.55), int(h*.45)+1),
                      int(w*.45):max(int(w*.55), int(w*.45)+1)] = cv2.GC_FGD
            bg_model, fg_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(small, grab_mask, None, bg_model, fg_model, 2, cv2.GC_INIT_WITH_MASK)
            segmented = np.where((grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
            segmented = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            regions, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for region in sorted(regions, key=cv2.contourArea, reverse=True)[:5]:
                if cv2.contourArea(region) < h * w * .025:
                    continue
                candidate = cv2.boxPoints(cv2.minAreaRect(region))
                score = _quad_quality(candidate, small, maps)
                if score > 0:
                    scored.append((score, candidate, "GrabCut foreground fallback"))
        except cv2.error:
            # The ordinary candidate pipeline remains usable if GrabCut cannot initialize.
            pass
    if not scored:
        return Detection(None, 0.0, "none", "No plausible document region was found. Place the four corners manually.")
    score, points, method = max(scored, key=lambda row: row[0])
    corners = order_points(points) / scale
    corners[:, 0] = np.clip(corners[:, 0], 0, oriented.width - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, oriented.height - 1)
    confidence = min(.97, max(0.0, score))
    message = "Review the proposed corners before applying the correction."
    if confidence < .62:
        message = "Automatic detection is uncertain. Review or adjust the corners, or switch to Manual mode."
    return Detection(corners.tolist(), confidence, method, message)


def _line_intersection(line_a, line_b):
    x1,y1,x2,y2 = line_a; x3,y3,x4,y4 = line_b
    den = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if abs(den) < 1e-6:
        return None
    determinant_a = x1*y2-y1*x2; determinant_b = x3*y4-y3*x4
    x = (determinant_a*(x3-x4)-(x1-x2)*determinant_b)/den
    y = (determinant_a*(y3-y4)-(y1-y2)*determinant_b)/den
    return [x,y]


def default_corners(size):
    width, height = size
    margin_x, margin_y = width * .08, height * .08
    return [[margin_x, margin_y], [width-1-margin_x, margin_y],
            [width-1-margin_x, height-1-margin_y], [margin_x, height-1-margin_y]]


def enhance_scan(image: Image.Image, mode: str = "Color") -> Image.Image:
    if mode == "Grayscale":
        return ImageOps.grayscale(image).convert("RGB")
    if mode == "Black & white":
        gray = ImageOps.grayscale(image)
        gray = ImageEnhance.Contrast(gray).enhance(1.8)
        return gray.point(lambda p: 255 if p > 165 else 0).convert("RGB")
    return image.convert("RGB")
