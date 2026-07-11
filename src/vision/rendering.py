# =============================================================
# rendering.py — rendering/crop saving helpers extracted from original vision.py
# =============================================================

import logging
import os
import re
from typing import List

import cv2
import numpy as np

from .geometry import detect_text_area

logger = logging.getLogger(__name__)

def draw_detections(image_bytes: bytes, pois: List[dict]) -> bytes:
    """Vẽ box đỏ, text và tọa độ pixel (x, y) lên ảnh (theo logic debug của USER)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes

    for p in pois:
        bbox = p.get("bbox")
        name = p.get("name", "")
        if not bbox:
            continue
        l, t, w, h = map(int, bbox)
        # Vẽ box ĐỎ (BGR: 0, 0, 255)
        cv2.rectangle(img, (l, t), (l + w, t + h), (0, 0, 255), 3)
        # Bỏ vẽ tên XANH (BGR: 255, 0, 0) để tránh rối mắt theo yêu cầu
        # cv2.putText(img, name, (l, max(t - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Vẽ điểm tâm và tọa độ pixel màu vàng (BGR: 0, 255, 255)
        cx = p.get("x")
        cy = p.get("y")
        if cx is not None and cy is not None:
            cx_i, cy_i = int(cx), int(cy)
            cv2.circle(img, (cx_i, cy_i), 6, (0, 255, 255), -1)  # Circle
            cv2.circle(img, (cx_i, cy_i), 7, (0, 0, 0), 1)       # Outline for contrast
            coord_str = f"({cx_i}, {cy_i})"
            cv2.putText(img, coord_str, (cx_i + 10, cy_i + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def enhance_for_detection(image_bytes: bytes) -> bytes:
    return image_bytes


def save_poi_crop(image_bytes: bytes, poi: dict, output_path: str, scale: float = 1.0):
    """Save text-only POI crop using bbox-relative component geometry."""
    try:
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        bbox = poi.get("bbox")
        if img is None or not bbox:
            return

        l, t, w, h = map(int, bbox)
        h_img, w_img = img.shape[:2]
        # Keep context slightly larger than a typical 24–28 px map icon, but avoid
        # pulling broad road polygons into the saved crop.
        outer_x = max(28, int(34 * scale))
        outer_y = max(10, int(12 * scale))
        x1, x2 = max(0, l - outer_x), min(w_img, l + w + outer_x)
        y1, y2 = max(0, t - outer_y), min(h_img, t + h + outer_y)
        raw = img[y1:y2, x1:x2]
        if raw.size == 0:
            return

        hp, wp = raw.shape[:2]
        bx1, bx2 = l - x1, l + w - x1
        by1, by2 = t - y1, t + h - y1
        corner = max(2, min(8, hp // 4, wp // 4))
        samples = np.concatenate((
            raw[:corner, :corner].reshape(-1, 3), raw[:corner, -corner:].reshape(-1, 3),
            raw[-corner:, :corner].reshape(-1, 3), raw[-corner:, -corner:].reshape(-1, 3),
        ))
        bg = np.median(samples, axis=0).astype(np.uint8)

        hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        bg_gray = float(cv2.cvtColor(bg.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0])
        candidate = (((hsv[:, :, 1] > 38) & (hsv[:, :, 2] > 55)) |
                     (np.abs(gray.astype(np.float32) - bg_gray) > 48)).astype(np.uint8) * 255
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        band = max(20, int(28 * scale))
        edge_rois = (
            (max(0, by1 - band), min(hp, by2 + band), max(0, bx1 - band), min(wp, bx1 + band)),
            (max(0, by1 - band), min(hp, by2 + band), max(0, bx2 - band), min(wp, bx2 + band)),
            (max(0, by1 - band), min(hp, by1 + band), max(0, bx1 - band), min(wp, bx2 + band)),
        )
        icon_mask = np.zeros((hp, wp), np.uint8)
        for ya, yb, xa, xb in edge_rois:
            roi = candidate[ya:yb, xa:xb]
            count, labels, stats, _ = cv2.connectedComponentsWithStats(roi, 8)
            for label in range(1, count):
                cx, cy, cw, ch, area = stats[label]
                if area < max(35, int(38 * scale * scale)):
                    continue
                aspect = cw / max(1, ch)
                fill = area / max(1, cw * ch)
                # Require icon-scale geometry and a center close to a YOLO boundary.
                # Small compact glyph groups can otherwise be mistaken for icons and
                # removed before text bounds are measured.
                compact = 0.65 <= aspect <= 1.55 and fill >= 0.34
                min_side = max(14, int(15 * scale))
                max_side = max(34, int(38 * scale))
                icon_sized = min_side <= cw <= max_side and min_side <= ch <= max_side
                center_x = xa + cx + cw / 2.0
                center_y = ya + cy + ch / 2.0
                near_vertical_edge = min(abs(center_x - bx1), abs(center_x - bx2)) <= max(14, int(18 * scale))
                near_top_edge = abs(center_y - by1) <= max(14, int(18 * scale))
                if compact and icon_sized and (near_vertical_edge or near_top_edge):
                    component = (labels == label).astype(np.uint8) * 255
                    icon_mask[ya:yb, xa:xb] |= component
        icon_mask = cv2.dilate(icon_mask, np.ones((5, 5), np.uint8), iterations=1)

        detect = raw.copy()
        detect[icon_mask > 0] = bg
        tx1, ty1, tx2, ty2 = detect_text_area(detect, scale)
        # Keep crop compact, but reserve extra left safety because antialiased first
        # glyphs are the most common part missed by projection-based text bounds.
        pad_l = max(10, int(14 * scale))
        pad_r = max(8, int(10 * scale))
        pad_t = max(4, int(5 * scale))
        pad_b = max(5, int(6 * scale))
        fx1, fy1 = max(0, tx1 - pad_l), max(0, ty1 - pad_t)
        fx2, fy2 = min(wp, tx2 + pad_r), min(hp, ty2 + pad_b)
        crop = raw[fy1:fy2, fx1:fx2].copy()
        if crop.size == 0:
            return

        # Protect detected text rectangle, erase compact icons only in its padding.
        local_mask = icon_mask[fy1:fy2, fx1:fx2]
        guard = np.zeros_like(local_mask)
        gx1, gx2 = max(0, tx1 - fx1), min(crop.shape[1], tx2 - fx1)
        gy1, gy2 = max(0, ty1 - fy1), min(crop.shape[0], ty2 - fy1)
        guard[gy1:gy2, gx1:gx2] = 255
        crop[(local_mask > 0) & (guard == 0)] = bg

        edge = max(1, int(scale))
        if crop.shape[0] > edge * 2:
            crop[:edge, :] = bg
            crop[-edge:, :] = bg

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ok, encoded = cv2.imencode(".png", crop)
        if ok:
            with open(output_path, "wb") as handle:
                handle.write(encoded.tobytes())
        else:
            cv2.imwrite(output_path, crop)
        logger.info("  [Crop saved] %s", os.path.basename(output_path))
    except Exception as exc:
        logger.warning("Không thể lưu ảnh crop POI: %s", exc)
