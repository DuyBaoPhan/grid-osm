# =============================================================
# detection.py — detection pipeline extracted from original vision.py
# =============================================================

import asyncio
import logging
import os
import re
from typing import List, Tuple

import cv2
import numpy as np

from config import SAVE_POI_CROPS, POI_CROPS_DIR
from .models import _get_yolo_model
from .text_cleaning import (
    _clean_ocr_edge_segments,
    _clean_spelling,
    _is_known_token,
    _looks_like_bad_ocr,
)
from .geometry import expand_bbox_downward, detect_text_area
from .recognizers import _recognize_text_crop_vietocr
from .rendering import save_poi_crop

logger = logging.getLogger(__name__)

def _should_drop_poi_name(name: str) -> bool:
    """True nếu tên POI sau OCR giống artifact hơn là tên địa điểm thật."""
    s = (name or "").strip()
    if not s:
        return True
    words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', s)
    if not words:
        return True
    if len(words) == 1:
        token = words[0]
        has_alpha = any(ch.isalpha() for ch in token)
        has_digit = any(ch.isdigit() for ch in token)
        has_viet = bool(re.search(r'[À-ỹĐđ]', token))
        # Single token mixed alpha+digit like `Con19` is usually an icon/count artifact.
        # Keep obvious uppercase/brand-like IDs (e.g. Q1, KFC, TP) and Vietnamese words.
        if has_alpha and has_digit and not has_viet and not token.isupper():
            return True
        if _looks_like_bad_ocr(token) and not _is_known_token(token):
            return True
    return False


def _merge_overlapping_boxes(boxes_list: List[dict], scale: float = 1.0) -> List[dict]:
    """
    Sát nhập các bounding box quá gần nhau hoặc chồng chập (như icon và nhãn văn bản của cùng một địa điểm).
    Giúp tránh trùng lặp POI và tránh mất thông tin khi nhãn bị tách đôi.
    """
    if len(boxes_list) <= 1:
        return boxes_list

    items = []
    for item in boxes_list:
        left, top, right, bottom = item["bbox"]
        items.append({
            "x1": left,
            "y1": top,
            "x2": right,
            "y2": bottom,
            "conf": item["confidence"],
        })

    merged_something = True
    while merged_something:
        merged_something = False
        n = len(items)
        i = 0
        while i < n:
            j = i + 1
            while j < n:
                item_a = items[i]
                item_b = items[j]

                cx_a = (item_a["x1"] + item_a["x2"]) / 2.0
                cy_a = (item_a["y1"] + item_a["y2"]) / 2.0
                cx_b = (item_b["x1"] + item_b["x2"]) / 2.0
                cy_b = (item_b["y1"] + item_b["y2"]) / 2.0

                dist_x = abs(cx_a - cx_b)
                dist_y = abs(cy_a - cy_b)
                dist = np.sqrt(dist_x**2 + dist_y**2)

                ix1 = max(item_a["x1"], item_b["x1"])
                iy1 = max(item_a["y1"], item_b["y1"])
                ix2 = min(item_a["x2"], item_b["x2"])
                iy2 = min(item_a["y2"], item_b["y2"])

                inter_w = max(0.0, ix2 - ix1)
                inter_h = max(0.0, iy2 - iy1)
                inter_area = inter_w * inter_h

                area_a = (item_a["x2"] - item_a["x1"]) * (item_a["y2"] - item_a["y1"])
                area_b = (item_b["x2"] - item_b["x1"]) * (item_b["y2"] - item_b["y1"])
                min_area = min(area_a, area_b)
                io_min = inter_area / min_area if min_area > 0 else 0.0

                should_merge = False
                if io_min > 0.65:
                    should_merge = True
                elif dist < 45.0 * scale:
                    if dist_x < 15.0 * scale or dist_y < 15.0 * scale:
                        should_merge = True

                if should_merge:
                    reason = "overlap" if io_min > 0.65 else "near-aligned"
                    merged_box = [
                        min(item_a["x1"], item_b["x1"]),
                        min(item_a["y1"], item_b["y1"]),
                        max(item_a["x2"], item_b["x2"]),
                        max(item_a["y2"], item_b["y2"]),
                    ]
                    logger.info(
                        "  [BBoxMerge] %s dist=%.1f dx=%.1f dy=%.1f io_min=%.2f "
                        "A=(%.0f,%.0f,%.0f,%.0f) B=(%.0f,%.0f,%.0f,%.0f) -> "
                        "(%.0f,%.0f,%.0f,%.0f)",
                        reason,
                        dist,
                        dist_x,
                        dist_y,
                        io_min,
                        item_a["x1"], item_a["y1"], item_a["x2"], item_a["y2"],
                        item_b["x1"], item_b["y1"], item_b["x2"], item_b["y2"],
                        merged_box[0], merged_box[1], merged_box[2], merged_box[3],
                    )
                    item_a["x1"] = merged_box[0]
                    item_a["y1"] = merged_box[1]
                    item_a["x2"] = merged_box[2]
                    item_a["y2"] = merged_box[3]
                    item_a["conf"] = max(item_a["conf"], item_b["conf"])
                    
                    items.pop(j)
                    n -= 1
                    merged_something = True
                    break
                j += 1
            if merged_something:
                break
            i += 1

    merged_boxes = [{
        "bbox": [item["x1"], item["y1"], item["x2"], item["y2"]],
        "confidence": item["conf"],
    } for item in items]
    if len(merged_boxes) != len(boxes_list):
        logger.info("  [BBoxMerge] Reduced detections %d -> %d", len(boxes_list), len(merged_boxes))
    return merged_boxes


async def extract_pois_from_screenshot(
    screenshot_bytes: bytes,
    tx: int = 0,
    ty: int = 0,
    img_metadata: dict = None,
) -> Tuple[List[dict], bool]:
    model = _get_yolo_model()
    if model is None:
        return [], False

    nparr = np.frombuffer(screenshot_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return [], False

    # Tự động tính toán scale factor (mật độ điểm ảnh)
    scale = 1.0
    if img_metadata and "scale" in img_metadata:
        scale = float(img_metadata["scale"])
    else:
        # Ước tính từ độ phân giải ảnh (1080px làm tiêu chuẩn)
        from config import SCREENSHOT_H
        scale = img.shape[0] / SCREENSHOT_H

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, lambda: model.predict(img, conf=0.2))
    
    pois = []
    if not results or len(results) == 0:
        return [], False

    result = results[0]
    boxes = result.boxes

    core_x1 = float(img_metadata.get("core_x1", 0.0)) if img_metadata else 0.0
    core_y1 = float(img_metadata.get("core_y1", 0.0)) if img_metadata else 0.0
    core_x2 = float(img_metadata.get("core_x2", img.shape[1])) if img_metadata else float(img.shape[1])
    core_y2 = float(img_metadata.get("core_y2", img.shape[0])) if img_metadata else float(img.shape[0])
    core_margin = max(8.0, 12.0 * scale)
    skipped_overlap = 0
    
    raw_detections = []
    for i, box in enumerate(boxes):
        b = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        
        left = float(b[0])
        top = float(b[1])
        right = float(b[2])
        bottom = float(b[3])
        box_cx = (left + right) / 2.0
        box_cy = (top + bottom) / 2.0
        if (
            box_cx < core_x1 - core_margin or box_cx > core_x2 + core_margin
            or box_cy < core_y1 - core_margin or box_cy > core_y2 + core_margin
        ):
            skipped_overlap += 1
            continue
            
        raw_detections.append({
            "bbox": [left, top, right, bottom],
            "confidence": conf,
        })

    # Sát nhập các detection quá gần nhau hoặc chồng chập
    merged_detections = _merge_overlapping_boxes(raw_detections, scale)
    if raw_detections or skipped_overlap:
        logger.info(
            "  [YOLO] tile=(%s,%s) raw=%d kept=%d merged=%d skipped_overlap=%d scale=%.2f",
            tx,
            ty,
            len(boxes),
            len(raw_detections),
            len(merged_detections),
            skipped_overlap,
            scale,
        )
    
    for i, det in enumerate(merged_detections):
        left, top, right, bottom = det["bbox"]
        conf = det["confidence"]
        height = bottom - top
        width = right - left
        box_cx = (left + right) / 2.0
        box_cy = (top + bottom) / 2.0
        
        # Crop squares for scores using a fixed icon size of 32 * scale
        icon_size_check = int(32 * scale)
        h_img, w_img = img.shape[:2]
        
        # Left square
        y1_l = max(0, int(top))
        y2_l = min(h_img, int(top + icon_size_check))
        x1_l = max(0, int(left))
        x2_l = min(w_img, int(left + icon_size_check))
        
        # Right square
        y1_r = max(0, int(top))
        y2_r = min(h_img, int(top + icon_size_check))
        x1_r = max(0, int(right - icon_size_check))
        x2_r = min(w_img, int(right))
        
        # Top square: lấy vùng vuông ở giữa-trên của bbox
        cx_center = (left + right) / 2.0
        x1_t = max(0, int(cx_center - icon_size_check / 2))
        x2_t = min(w_img, int(cx_center + icon_size_check / 2))
        y1_t = max(0, int(top))
        y2_t = min(h_img, int(top + icon_size_check))
        
        left_sq = img[y1_l:y2_l, x1_l:x2_l]
        right_sq = img[y1_r:y2_r, x1_r:x2_r]
        top_sq = img[y1_t:y2_t, x1_t:x2_t]
        
        def get_square_score(sq):
            if sq is None or sq.size == 0:
                return -999.0
            try:
                hsv = cv2.cvtColor(sq, cv2.COLOR_BGR2HSV)
                s = hsv[:, :, 1]
                v = hsv[:, :, 2]
                
                fg_mask = ~((v > 215) & (s < 30))
                
                h_sz, w_sz = sq.shape[:2]
                cy_min, cy_max = int(0.25 * h_sz), int(0.75 * h_sz)
                cx_min, cx_max = int(0.25 * w_sz), int(0.75 * w_sz)
                center_mask = fg_mask[cy_min:cy_max, cx_min:cx_max]
                center_fg_ratio = np.mean(center_mask) if center_mask.size > 0 else 0.0
                
                mean_sat = float(np.mean(s))
                mean_val = float(np.mean(v))
                
                score = mean_sat + 60.0 * center_fg_ratio - 0.2 * mean_val
                return score
            except Exception:
                return -999.0
                
        score_left = get_square_score(left_sq)
        score_right = get_square_score(right_sq)
        score_top = get_square_score(top_sq)
        
        THRESHOLD = 10.0
        # Nếu không có icon rõ ràng ở bất kỳ phía nào → mặc định là left (hầu hết POI đều có icon bên trái)
        # để bắt được cả các icon màu xám/nhạt không vượt qua THRESHOLD
        if max(score_left, score_right, score_top) < THRESHOLD:
            icon_side = "left"
        elif score_top > score_left + THRESHOLD and score_top > score_right + THRESHOLD:
            icon_side = "top"
        elif score_right > score_left + THRESHOLD:
            icon_side = "right"
        else:
            icon_side = "left"

        # Định vị chính xác tâm icon bằng phương pháp tìm trọng tâm (centroid) màu sắc
        icon_w_search = int(30 * scale)
        icon_h_search = int(30 * scale)
        h_img, w_img = img.shape[:2]
        
        if icon_side == "left":
            x1_s = max(0, int(left))
            x2_s = min(w_img, int(left + icon_w_search))
            y1_s = max(0, int(top))
            y2_s = min(h_img, int(top + icon_h_search))
        elif icon_side == "right":
            x1_s = max(0, int(right - icon_w_search))
            x2_s = min(w_img, int(right))
            y1_s = max(0, int(top))
            y2_s = min(h_img, int(top + icon_h_search))
        elif icon_side == "top":
            cx_center = (left + right) / 2.0
            x1_s = max(0, int(cx_center - icon_w_search / 2))
            x2_s = min(w_img, int(cx_center + icon_w_search / 2))
            y1_s = max(0, int(top))
            y2_s = min(h_img, int(top + icon_h_search))
        else:
            x1_s = y1_s = x2_s = y2_s = 0

        cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
        if x2_s > x1_s and y2_s > y1_s:
            search_area = img[y1_s:y2_s, x1_s:x2_s]
            hsv = cv2.cvtColor(search_area, cv2.COLOR_BGR2HSV)
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
            
            # Ngưỡng màu để lọc ra biểu tượng có màu sắc của Google Maps
            SAT_THRESH = 35
            VAL_MIN = 60
            VAL_MAX = 256
            
            icon_mask = (s > SAT_THRESH) & (v > VAL_MIN) & (v < VAL_MAX)
            pixel_count = np.sum(icon_mask)
            
            if pixel_count >= int(15 * scale * scale):
                ys, xs = np.where(icon_mask)
                cx = x1_s + np.mean(xs)
                cy = y1_s + np.mean(ys)
            else:
                # Fallback nếu không có đủ pixel màu (icon xám)
                cx = x1_s + (x2_s - x1_s) / 2.0
                cy = y1_s + (y2_s - y1_s) / 2.0
        
        # Mở rộng bbox xuống dưới để ôm trọn dòng chữ thứ 2 (hoặc 3)
        left_exp, top_exp, right_exp, bottom_exp = expand_bbox_downward(img, [left, top, right, bottom], max_expand=int(28 * scale))
        b_expanded = [left_exp, top_exp, right_exp, bottom_exp]
        
        # OCR text dùng logic phóng to 6x và phân tích đa dòng (loại bỏ icon)
        text = _recognize_text_crop_vietocr(img, b_expanded, icon_side, scale, cx=cx, cy=cy)
        name = _clean_spelling(text)
        name_cleaned = _clean_ocr_edge_segments(name)
        if name_cleaned != name:
            logger.info("  [OCR edge cleanup] '%s' -> '%s'", name, name_cleaned)
            name = name_cleaned
        logger.debug(
            "  [POI OCR] tile=(%s,%s) idx=%d conf=%.3f icon=%s box=(%.0f,%.0f,%.0f,%.0f) "
            "expanded=(%.0f,%.0f,%.0f,%.0f) center=(%.1f,%.1f) text='%s' name='%s'",
            tx,
            ty,
            i,
            conf,
            icon_side,
            left,
            top,
            right,
            bottom,
            left_exp,
            top_exp,
            right_exp,
            bottom_exp,
            cx,
            cy,
            text,
            name,
        )
        
        if _should_drop_poi_name(name):
            logger.info(
                "  [POI filtered] tile=(%s,%s) idx=%d name='%s' conf=%.3f box=(%.0f,%.0f,%.0f,%.0f)",
                tx,
                ty,
                i,
                name,
                conf,
                left,
                top,
                right,
                bottom,
            )
            continue
        
        if not name:
            name = f"Unknown_{i}"
  
        # Đánh dấu POI bị cắt sát mép ảnh để worker chụp rescue view riêng.
        edge_margin = max(24.0, 48.0 * scale)
        edge_sides = []
        if left_exp <= edge_margin:
            edge_sides.append("left")
        if right_exp >= (w_img - edge_margin):
            edge_sides.append("right")
        if top_exp <= edge_margin:
            edge_sides.append("top")
        if bottom_exp >= (h_img - edge_margin):
            edge_sides.append("bottom")

        poi_item = {
            "name": name,
            "x": cx,
            "y": cy,
            "confidence": conf,
            "bbox": [float(left_exp), float(top_exp), float(right_exp - left_exp), float(bottom_exp - top_exp)],
            "has_icon": True,
            "icon_side": icon_side,
            "yolo_height": float(height),
            "edge_cut": bool(edge_sides),
            "edge_sides": edge_sides,
            "edge_margin_px": float(edge_margin),
        }
        pois.append(poi_item)
  
        if SAVE_POI_CROPS:
            # Làm sạch tên file để loại bỏ ký tự không hợp lệ trên Windows
            safe_name = re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_").strip()
            if not safe_name:
                safe_name = f"Unknown_{i}"
            filename = f"tile_{tx}_{ty}_poi_{i}_{safe_name}.png"
            output_path = os.path.join(POI_CROPS_DIR, filename)
            save_poi_crop(screenshot_bytes, poi_item, output_path, scale)
            logger.debug("  [Crop saved] %s", output_path)
            
    if skipped_overlap:
        logger.info("  [OverlapFilter] Skipped %d detections outside core tile before OCR", skipped_overlap)

    return pois, False
