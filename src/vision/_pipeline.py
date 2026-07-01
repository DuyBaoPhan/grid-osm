# Compatibility shim. Detection pipeline moved to detection.py.

from .detection import (
    _should_drop_poi_name,
    _merge_overlapping_boxes,
    extract_pois_from_screenshot,
)

__all__ = [
    _should_drop_poi_name,
    _merge_overlapping_boxes,
    extract_pois_from_screenshot,
]
