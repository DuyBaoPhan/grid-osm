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
    """Lưu ảnh crop POI đẹp: chỉ text, padding đều, không lộ icon, không cắt chữ."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return

        bbox = poi.get("bbox")
        if not bbox:
            return

        l, t, w, h = map(int, bbox)
        h_img, w_img = img.shape[:2]

        # Lấy vùng rộng hơn bbox để detect đủ chữ sát biên trước khi crop lại đẹp.
        outer_pad = max(10, int(12 * scale))
        y1 = max(0, t - outer_pad)
        y2 = min(h_img, t + h + outer_pad)
        x1 = max(0, l - outer_pad)
        x2 = min(w_img, l + w + outer_pad)

        raw_crop = img[y1:y2, x1:x2]
        if raw_crop.size == 0:
            return

        bg_color = np.median(raw_crop, axis=(0, 1)).astype(int).tolist()
        h_rc, w_rc = raw_crop.shape[:2]

        cx = poi.get("x")
        cy = poi.get("y")
        icon_side = poi.get("icon_side", "left")
        cx_local = (cx - x1) if cx is not None else None
        cy_local = (cy - y1) if cy is not None else None

        def _clean_outer_border(img_part: np.ndarray):
            """Dọn viền của ảnh detect để không bắt nhiễu map ở vùng lấy rộng."""
            border_w = max(8, int(10 * scale))
            hp, wp = img_part.shape[:2]
            if border_w > 0:
                if border_w < hp:
                    img_part[0:border_w, :] = bg_color
                    img_part[hp - border_w:, :] = bg_color
                if border_w < wp:
                    img_part[:, 0:border_w] = bg_color
                    img_part[:, wp - border_w:] = bg_color

        def _mask_icon_for_detection(img_part: np.ndarray):
            """Mask icon chỉ để tìm text box, không dùng ảnh này làm crop cuối."""
            if img_part is None or img_part.size == 0 or cx_local is None or cy_local is None:
                return
            hp, wp = img_part.shape[:2]
            if icon_side == "left":
                mask_w = max(0, min(wp, int(cx_local + 7.5 * scale)))
                img_part[:, :mask_w] = bg_color
            elif icon_side == "right":
                mask_x = max(0, min(wp, int(cx_local - 7.5 * scale)))
                img_part[:, mask_x:] = bg_color
            elif icon_side == "top":
                mask_h = max(0, min(hp, int(cy_local + 7.5 * scale)))
                img_part[:mask_h, :] = bg_color

        # Ảnh detect có thể bị mask icon; crop cuối lấy từ raw_crop để không mất nét chữ.
        detect_img = raw_crop.copy()
        _clean_outer_border(detect_img)
        _mask_icon_for_detection(detect_img)

        tx1, ty1, tx2, ty2 = detect_text_area(detect_img, scale)

        # Padding đẹp 4 phía. Giữ phải rộng hơn vì chữ cuối thường sát biên/nhỏ.
        pad_left = max(6, int(8 * scale))
        pad_right = max(10, int(14 * scale))
        pad_top = max(4, int(5 * scale))
        pad_bottom = max(5, int(6 * scale))
        raw_tx1 = tx1
        raw_ty1 = ty1
        raw_tx2 = tx2
        raw_ty2 = ty2
        tx1 = max(0, raw_tx1 - pad_left)
        ty1 = max(0, raw_ty1 - pad_top)
        tx2 = min(w_rc, raw_tx2 + pad_right)
        ty2 = min(h_rc, raw_ty2 + pad_bottom)

        crop = raw_crop[ty1:ty2, tx1:tx2].copy()
        if crop.size == 0:
            return

        # Dọn icon sót trong vùng padding, không tô vào vùng text đã detect.
        final_h, final_w = crop.shape[:2]
        text_left_in_crop = max(0, raw_tx1 - tx1)
        text_right_in_crop = min(final_w, raw_tx2 - tx1)
        text_top_in_crop = max(0, raw_ty1 - ty1)

        if cx_local is not None and cy_local is not None:
            # Dọn vùng icon theo tọa độ thật, gồm cả icon trắng/bóng mờ low-saturation.
            # HSV-only không bắt được các mảng trắng/xám nên vẫn còn dấu vết.
            if icon_side == "left":
                icon_edge_in_crop = int(cx_local + 12.5 * scale) - tx1
                clean_w = max(0, min(final_w, icon_edge_in_crop))
                if clean_w > 0:
                    crop[:, :clean_w] = bg_color
            elif icon_side == "right":
                icon_edge_in_crop = int(cx_local - 12.5 * scale) - tx1
                clean_x = max(0, min(final_w, icon_edge_in_crop))
                if clean_x < final_w:
                    crop[:, clean_x:] = bg_color
            elif icon_side == "top":
                icon_edge_in_crop = int(cy_local + 12.5 * scale) - ty1
                clean_h = max(0, min(final_h, icon_edge_in_crop))
                if clean_h > 0:
                    crop[:clean_h, :] = bg_color

            # Dọn thêm mảng màu icon nếu còn chạm mép ngoài vùng hình chữ nhật.
            def _erase_edge_connected_icon(side: str):
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                s = hsv_crop[:, :, 1]
                v = hsv_crop[:, :, 2]
                icon_mask = ((s > 35) & (v > 60)).astype(np.uint8) * 255
                kernel = np.ones((3, 3), np.uint8)
                icon_mask = cv2.dilate(icon_mask, kernel, iterations=1)
                scan_w = min(final_w, max(18, int(28 * scale)))
                scan_h = min(final_h, max(18, int(28 * scale)))
                edge_tol = max(2, int(3 * scale))
                clean_mask = np.zeros((final_h, final_w), dtype=np.uint8)

                if side == "left":
                    roi = icon_mask[:, :scan_w]
                    x_base, y_base = 0, 0
                elif side == "right":
                    roi = icon_mask[:, final_w - scan_w:]
                    x_base, y_base = final_w - scan_w, 0
                elif side == "top":
                    roi = icon_mask[:scan_h, :]
                    x_base, y_base = 0, 0
                else:
                    return

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi, 8)
                for label in range(1, num_labels):
                    x, y, w_box, h_box, area = stats[label]
                    if area < max(2, int(3 * scale * scale)):
                        continue
                    touches_edge = (
                        (side == "left" and x <= edge_tol) or
                        (side == "right" and x + w_box >= roi.shape[1] - edge_tol) or
                        (side == "top" and y <= edge_tol)
                    )
                    if touches_edge:
                        component = (labels == label).astype(np.uint8) * 255
                        clean_mask[y_base:y_base + roi.shape[0], x_base:x_base + roi.shape[1]] |= component

                if np.any(clean_mask):
                    clean_mask[:] = cv2.dilate(clean_mask, kernel, iterations=1)
                    crop[clean_mask > 0] = bg_color

            if icon_side == "left":
                _erase_edge_connected_icon("left")
            elif icon_side == "right":
                _erase_edge_connected_icon("right")
            elif icon_side == "top":
                _erase_edge_connected_icon("top")

        # Làm sạch nền viền rất mỏng trên/dưới; không tô trái/phải để tránh mất nét đầu/cuối.
        edge_pad = max(1, int(1 * scale))
        if crop.shape[0] > 2 * edge_pad:
            crop[0:edge_pad, :] = bg_color
            crop[crop.shape[0] - edge_pad:, :] = bg_color

        def _remove_thin_colored_horizontal_lines(img_part: np.ndarray):
            """Xóa line xanh/cyan mảnh do Google Maps overlay/hover lọt vào ảnh crop lưu debug."""
            if img_part is None or img_part.size == 0:
                return img_part
            try:
                hp, wp = img_part.shape[:2]
                hsv = cv2.cvtColor(img_part, cv2.COLOR_BGR2HSV)
                h = hsv[:, :, 0]
                s = hsv[:, :, 1]
                v = hsv[:, :, 2]
                # Bắt line xanh/cyan bão hòa cao, rất mảnh và kéo dài ngang.
                line_mask = (((h >= 80) & (h <= 115) & (s > 45) & (v > 120))).astype(np.uint8) * 255
                kernel = np.ones((1, max(8, int(10 * scale))), np.uint8)
                line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_mask, 8)
                clean_mask = np.zeros((hp, wp), dtype=np.uint8)
                for label in range(1, num_labels):
                    x, y, w_box, h_box, area = stats[label]
                    if h_box <= max(3, int(3 * scale)) and w_box >= max(24, int(0.28 * wp)):
                        clean_mask[labels == label] = 255
                if np.any(clean_mask):
                    clean_mask = cv2.dilate(clean_mask, np.ones((2, 2), np.uint8), iterations=1)
                    img_part[clean_mask > 0] = bg_color
            except Exception:
                pass
            return img_part

        crop = _remove_thin_colored_horizontal_lines(crop)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        is_success, im_buf_arr = cv2.imencode(".png", crop)
        if is_success:
            with open(output_path, "wb") as f:
                f.write(im_buf_arr.tobytes())
        else:
            cv2.imwrite(output_path, crop)
        logger.info("  [Crop saved] %s", os.path.basename(output_path))
    except Exception as e:
        logger.warning("Không thể lưu ảnh crop POI: %s", e)
