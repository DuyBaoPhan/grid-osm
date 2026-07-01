# =============================================================
# models.py — model lazy-load helpers extracted from original vision.py
# =============================================================

import logging
import os

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
    PADDLE_TEXT_DET_ENABLED,
)

logger = logging.getLogger(__name__)

_YOLO_MODEL = None
_VIETOCR_PREDICTOR = None
_PADDLE_TEXT_DETECTOR = None

def _get_paddle_text_detector():
    """Lazy-load PaddleOCR để chỉ detect/recognize text."""
    global _PADDLE_TEXT_DETECTOR
    if not PADDLE_TEXT_DET_ENABLED:
        return None
    if _PADDLE_TEXT_DETECTOR is False:
        return None
    if _PADDLE_TEXT_DETECTOR is not None:
        return _PADDLE_TEXT_DETECTOR
    try:
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("FLAGS_enable_pir_api", "0")
        from paddleocr import PaddleOCR
        # Không truyền det/rec/cls ở constructor: một số bản PaddleOCR báo "Unknown argument: det".
        _PADDLE_TEXT_DETECTOR = PaddleOCR(
            use_angle_cls=False,
            lang="en",
        )
        logger.info("Loaded PaddleOCR detector")
        return _PADDLE_TEXT_DETECTOR
    except Exception as exc:
        _PADDLE_TEXT_DETECTOR = False
        logger.warning("Không load được PaddleOCR: %s", exc)
        return None


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
