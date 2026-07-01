# =============================================================
# crop_processing.py — crop preprocessing helpers extracted from original vision.py
# =============================================================

import logging
from typing import List

import cv2
import numpy as np

from .text_cleaning import _clean_ocr_edge_segments

logger = logging.getLogger(__name__)

def split_crop_into_lines(crop_img: np.ndarray, scale: float = 1.0) -> List[np.ndarray]:
    """
    Phân tích hình ảnh crop của POI và cắt thành các dòng chữ đơn lẻ (nếu có nhiều dòng).
    Trả về danh sách các sub-image tương ứng với từng dòng chữ.
    """
    h_sz, w_sz = crop_img.shape[:2]
    if h_sz < int(16 * scale):
        return [crop_img]
        
    try:
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        
        bg_s = np.median(s)
        bg_v = np.median(v)
        is_white_bg = (bg_s < 25)
        
        row_bg_ratios = []
        for y in range(h_sz):
            row_s = s[y, :]
            row_v = v[y, :]
            
            if is_white_bg:
                bg_pixels = np.sum((row_s < 35) & (row_v > 200))
            else:
                bg_pixels = np.sum((np.abs(row_s.astype(int) - int(bg_s)) < 40) & (np.abs(row_v.astype(int) - int(bg_v)) < 40))
                
            row_bg_ratios.append(bg_pixels / w_sz)
            
        row_bg_ratios = np.array(row_bg_ratios)
        
        # Tìm khoảng trống giữa các dòng (tỷ lệ nền > 93%)
        GAP_THRESH = 0.93
        is_gap = row_bg_ratios > GAP_THRESH
        
        bands = []
        in_band = False
        start_y = 0
        
        for y in range(h_sz):
            if not is_gap[y]:
                if not in_band:
                    start_y = y
                    in_band = True
            else:
                if in_band:
                    end_y = y
                    if end_y - start_y >= int(6 * scale):
                        bands.append((start_y, end_y))
                    in_band = False
                    
        if in_band:
            if h_sz - start_y >= int(6 * scale):
                bands.append((start_y, h_sz))
                
        if len(bands) <= 1:
            return [crop_img]
            
        line_crops = []
        pad_y = max(4, int(4 * scale))
        for sy, ey in bands:
            y1 = max(0, sy - pad_y)
            y2 = min(h_sz, ey + pad_y)
            line_crops.append(crop_img[y1:y2, :])
            
        return line_crops
        
    except Exception as e:
        logger.debug("Lỗi khi chia dòng OCR: %s", e)
        return [crop_img]


def normalize_ocr_background(crop_img: np.ndarray) -> np.ndarray:
    """Chuẩn hóa nền crop sang màu ngà sáng, giữ chữ màu/đậm để OCR đọc ổn định hơn."""
    if crop_img is None or crop_img.size == 0:
        return crop_img
    try:
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # Giữ chữ màu đậm (V < 165) hoặc chữ có màu sắc sặc sỡ (S > 80)
        # Loại bỏ các vùng nền xám/xanh nhạt của đường và nền đất ngà
        text_mask = (v < 165) | (s > 80)
        text_mask = cv2.dilate(text_mask.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1).astype(bool)
        clean_bg = np.array([245, 242, 232], dtype=np.uint8)  # BGR ngà nhạt giống crop đọc tốt

        out = crop_img.copy()
        out[~text_mask] = clean_bg
        return out
    except Exception:
        return crop_img


def split_crop_into_lines_normalized(crop_img: np.ndarray, scale: float = 1.0) -> List[np.ndarray]:
    """Tách crop thành các dòng chữ đơn lẻ dùng ngưỡng chiều cao tối thiểu 3*scale."""
    h_sz, w_sz = crop_img.shape[:2]
    if h_sz < int(12 * scale):
        return [crop_img]
        
    try:
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        
        bg_s = np.median(s)
        bg_v = np.median(v)
        is_white_bg = (bg_s < 25)
        
        row_bg_ratios = []
        for y in range(h_sz):
            row_s = s[y, :]
            row_v = v[y, :]
            
            if is_white_bg:
                bg_pixels = np.sum((row_s < 35) & (row_v > 200))
            else:
                bg_pixels = np.sum((np.abs(row_s.astype(int) - int(bg_s)) < 40) & (np.abs(row_v.astype(int) - int(bg_v)) < 40))
                
            row_bg_ratios.append(bg_pixels / w_sz)
            
        row_bg_ratios = np.array(row_bg_ratios)
        
        GAP_THRESH = 0.93
        is_gap = row_bg_ratios > GAP_THRESH
        
        bands = []
        in_band = False
        start_y = 0
        
        for y in range(h_sz):
            if not is_gap[y]:
                if not in_band:
                    start_y = y
                    in_band = True
            else:
                if in_band:
                    end_y = y
                    if end_y - start_y >= int(3 * scale):
                        bands.append((start_y, end_y))
                    in_band = False
                    
        if in_band:
            if h_sz - start_y >= int(3 * scale):
                bands.append((start_y, h_sz))
                
        if len(bands) <= 1:
            return [crop_img]
            
        line_crops = []
        pad_y = max(4, int(4 * scale))
        for sy, ey in bands:
            y1 = max(0, sy - pad_y)
            y2 = min(h_sz, ey + pad_y)
            line_crops.append(crop_img[y1:y2, :])
            
        return line_crops
        
    except Exception as e:
        logger.debug("Lỗi khi chia dòng OCR normalized: %s", e)
        return [crop_img]


def _select_primary_line_crops(line_crops: List[np.ndarray]) -> List[np.ndarray]:
    """Chỉ giữ dòng tên chính: ưu tiên dòng có font/chiều cao chữ lớn nhất, bỏ mô tả nhỏ bên dưới."""
    usable = [crop for crop in line_crops if crop is not None and crop.size > 0]
    if len(usable) <= 1:
        return usable

    heights = [crop.shape[0] for crop in usable]
    max_h = max(heights)
    if max_h <= 0:
        return usable[:1]

    # Google Maps thường đặt tên chính ở đầu nhãn; category/mô tả nằm dưới.
    # Padding làm dòng nhỏ có crop height gần dòng chính, nên không chọn mọi dòng gần max.
    first_h = heights[0]
    keep_threshold = max(first_h * 0.92, max_h * 0.82)
    kept = []
    for crop, h in zip(usable, heights):
        if h >= keep_threshold:
            kept.append(crop)
            continue
        break

    return kept or [usable[0]]
