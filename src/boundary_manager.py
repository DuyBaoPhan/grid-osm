from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import config


def slugify(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("đ", "d")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unknown"


def boundaries_dir() -> Path:
    return Path(getattr(config, "BOUNDARIES_DIR", config.BASE_DIR / "data" / "boundaries"))


def manifest_path() -> Path:
    return Path(getattr(config, "BOUNDARY_MANIFEST_FILE", boundaries_dir() / "manifest.json"))


def boundary_path(province: str, district: str) -> Path:
    return boundaries_dir() / slugify(province) / f"{slugify(district)}.geojson"


def load_geojson(path: str | Path) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("type") == "Feature":
        data = data.get("geometry") or {}
    if data.get("type") in {"Polygon", "MultiPolygon"} and data.get("coordinates"):
        return data
    return None


def extract_points(geometry: dict) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    def walk(value: object) -> None:
        if (
            isinstance(value, list)
            and len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            points.append((float(value[1]), float(value[0])))
            return
        if isinstance(value, list):
            for item in value:
                walk(item)

    walk(geometry.get("coordinates", []))
    return points


def geometry_bbox(geometry: dict) -> Optional[list[float]]:
    points = extract_points(geometry)
    if not points:
        return None
    lats = [p[0] for p in points]
    lngs = [p[1] for p in points]
    return [min(lats), min(lngs), max(lats), max(lngs)]


def geometry_centroid(geometry: dict) -> Optional[list[float]]:
    bbox = geometry_bbox(geometry)
    if not bbox:
        return None
    lat_min, lng_min, lat_max, lng_max = bbox
    return [(lat_min + lat_max) / 2.0, (lng_min + lng_max) / 2.0]


def read_manifest(path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path else manifest_path()
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("areas", [])
    else:
        rows = data
    return [row for row in rows if isinstance(row, dict)]


def write_manifest(rows: Iterable[dict], path: str | Path | None = None) -> Path:
    p = Path(path) if path else manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"areas": list(rows)}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
