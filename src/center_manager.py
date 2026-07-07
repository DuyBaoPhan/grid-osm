from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import config
from boundary_manager import slugify


def centers_dir() -> Path:
    return Path(getattr(config, "CENTERS_DIR", config.BASE_DIR / "centers"))


def center_points_path() -> Path:
    return Path(getattr(config, "CENTER_POINTS_FILE", centers_dir() / "post_office_centers.json"))


def center_key(province: str, district: str) -> str:
    return f"{slugify(province)}/{slugify(district)}"


def read_centers(path: str | Path | None = None) -> dict[str, dict]:
    p = Path(path) if path else center_points_path()
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = data.get("centers", {}) if isinstance(data, dict) else {}
    return rows if isinstance(rows, dict) else {}


def write_centers(rows: dict[str, dict], path: str | Path | None = None) -> Path:
    p = Path(path) if path else center_points_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"centers": rows}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def get_center(province: str, district: str, centers: dict[str, dict] | None = None) -> dict | None:
    data = centers if centers is not None else read_centers()
    row = data.get(center_key(province, district))
    return row if isinstance(row, dict) else None
