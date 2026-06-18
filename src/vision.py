# =============================================================
# vision.py — Nhận diện POI bằng YOLOv8 & VietOCR (Code chuẩn của USER)
# =============================================================

import asyncio
import io
import logging
import os
import re
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

# Patch torch.load for PyTorch 2.6 compatibility before importing ultralytics/yolo
try:
    import torch
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load
except ImportError:
    pass

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from config import (
    YOLO_MODEL_PATH,
    VIETOCR_MODEL,
    VIETOCR_DEVICE,
    SAVE_POI_CROPS,
    POI_CROPS_DIR,
)

logger = logging.getLogger(__name__)

_YOLO_MODEL = None
_VIETOCR_PREDICTOR = None

def _get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        if YOLO is None:
            logger.error("Thư viện 'ultralytics' chưa được cài đặt.")
            return None
        try:
            _YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
            logger.info("Loaded YOLO model from %s", YOLO_MODEL_PATH)
        except Exception as e:
            logger.error("Không thể load YOLO model: %s", e)
    return _YOLO_MODEL

def _get_vietocr_predictor():
    global _VIETOCR_PREDICTOR
    if _VIETOCR_PREDICTOR is not None:
        return _VIETOCR_PREDICTOR
    try:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
        config = Cfg.load_config_from_name(VIETOCR_MODEL)
        config["device"] = VIETOCR_DEVICE
        config["predictor"]["beamsearch"] = True
        # Theo code của USER: cnn pretrained = False
        if "cnn" in config:
            config["cnn"]["pretrained"] = False
        _VIETOCR_PREDICTOR = Predictor(config)
        logger.info("Loaded VietOCR model=%s", VIETOCR_MODEL)
        return _VIETOCR_PREDICTOR
    except Exception as exc:
        logger.warning("Không load được VietOCR: %s", exc)
        return None

def _strip_vietnamese_accents(s: str) -> str:
    """Loại bỏ hoàn toàn dấu tiếng Việt và đưa về chữ thường (Dùng cho deduplicate)."""
    s = s.lower()
    s = re.sub(r"[àáảãạăằắẳẵặâầấẩẫậ]", "a", s)
    s = re.sub(r"[èéẻẽẹêềếểễệ]", "e", s)
    s = re.sub(r"[ìíỉĩị]", "i", s)
    s = re.sub(r"[òóỏõọôồốổỗộơờớởỡợ]", "o", s)
    s = re.sub(r"[ùúủũụưừứửữự]", "u", s)
    s = re.sub(r"[ỳýỷỹỵ]", "y", s)
    s = re.sub(r"[đ]", "d", s)
    return s

def _clean_spelling(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _recognize_text_crop_vietocr(cv_img: np.ndarray, bbox: List[float]) -> str:
    predictor = _get_vietocr_predictor()
    if predictor is None or cv_img is None:
        return ""

    h_img, w_img = cv_img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    
    # 1. Enlarge bbox (Padding 10px theo code USER)
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w_img, x2 + pad)
    y2 = min(h_img, y2 + pad)

    crop = cv_img[y1:y2, x1:x2]
    if crop.size == 0:
        return ""

    try:
        # 2. Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # 3. Resize 6x (Theo code USER)
        gray = cv2.resize(
            gray,
            None,
            fx=6,
            fy=6,
            interpolation=cv2.INTER_CUBIC
        )
        
        # 4. Denoise nhẹ bằng GaussianBlur
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 5. Predict
        pil_img = Image.fromarray(gray)
        text = predictor.predict(pil_img)
        return (text or "").strip()
    except Exception as e:
        logger.debug("VietOCR error: %s", e)
        return ""

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

    loop = asyncio.get_event_loop()
    # Hạ conf xuống 0.2 để nhạy hơn (nãy 0.5 là quá cao)
    results = await loop.run_in_executor(None, lambda: model.predict(img, conf=0.2))
    
    pois = []
    if not results or len(results) == 0:
        return [], False

    result = results[0]
    boxes = result.boxes
    
    for i, box in enumerate(boxes):
        b = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        
        # Tâm pixel để tính GPS
        cx = float((b[0] + b[2]) / 2.0)
        cy = float((b[1] + b[3]) / 2.0)
        
        # OCR text dùng logic phóng to 6x
        text = _recognize_text_crop_vietocr(img, b.tolist())
        name = _clean_spelling(text)
        
        if not name:
            name = f"Unknown_{i}"

        poi_item = {
            "name": name,
            "x": cx,
            "y": cy,
            "confidence": conf,
            "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
            "has_icon": True
        }
        pois.append(poi_item)

        if SAVE_POI_CROPS:
            # Làm sạch tên file để loại bỏ ký tự không hợp lệ trên Windows
            safe_name = re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_").strip()
            if not safe_name:
                safe_name = f"Unknown_{i}"
            filename = f"tile_{tx}_{ty}_poi_{i}_{safe_name}.png"
            output_path = os.path.join(POI_CROPS_DIR, filename)
            save_poi_crop(screenshot_bytes, poi_item, output_path)

    return pois, False

def draw_detections(image_bytes: bytes, pois: List[dict]) -> bytes:
    """Vẽ box đỏ và text lên ảnh (theo logic debug của USER)."""
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
        # Vẽ tên XANH (BGR: 255, 0, 0)
        cv2.putText(img, name, (l, max(t - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()

def enhance_for_detection(image_bytes: bytes) -> bytes:
    return image_bytes

def remove_background(image_bytes: bytes) -> bytes:
    return image_bytes

def save_poi_crop(image_bytes: bytes, poi: dict, output_path: str, crop_x: int = 0, crop_y: int = 0):
    """Lưu ảnh crop của POI."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return
        
        bbox = poi.get("bbox")
        if not bbox:
            return
        
        # bbox có dạng [left, top, width, height]
        l, t, w, h = map(int, bbox)
        h_img, w_img = img.shape[:2]
        
        # Thêm 5 pixel padding xung quanh POI để ảnh crop rõ ràng hơn
        pad = 5
        x1 = max(0, l - pad)
        y1 = max(0, t - pad)
        x2 = min(w_img, l + w + pad)
        y2 = min(h_img, t + h + pad)
        
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, crop)
            logger.info("  [Crop saved] %s", os.path.basename(output_path))
    except Exception as e:
        logger.warning("Không thể lưu ảnh crop POI: %s", e)
