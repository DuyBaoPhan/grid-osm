import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from playwright.async_api import async_playwright

import config
from config import LOG_FILE, LOG_LEVEL, NUM_WORKERS
from src.coordinator import Coordinator
from src.scan_context import ScanArea, get_single_config_area, iter_manifest_areas
from src.worker import Worker


def _setup_logging() -> None:
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%H:%M:%S"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


logger = logging.getLogger(__name__)
MAP_SERVER_PORT = 8765
_http_server: ThreadingHTTPServer | None = None


def _start_map_server(directory: str, port: int) -> ThreadingHTTPServer:
    class _Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Google Maps POI scraper")
    parser.add_argument("--scan-mode", choices=["single", "manifest"], default=config.SCAN_MODE)
    parser.add_argument("--province", default=config.SCAN_AREA_FILTER_PROVINCE)
    parser.add_argument("--district", default=config.SCAN_AREA_FILTER_DISTRICT)
    parser.add_argument("--start-index", type=int, default=config.SCAN_START_INDEX)
    parser.add_argument("--max-areas", type=int, default=config.SCAN_MAX_AREAS)
    return parser.parse_args()


def _areas_from_args(args: argparse.Namespace) -> list[ScanArea]:
    if args.scan_mode == "manifest":
        return list(iter_manifest_areas(args.province, args.district, args.start_index, args.max_areas))
    return [get_single_config_area()]


async def _run_area(area: ScanArea, pw) -> None:
    logger.info("=" * 60)
    logger.info("  Area: %s / %s", area.province, area.district)
    logger.info("  Runtime: %s", area.runtime_dir)
    logger.info("=" * 60)

    coord = Coordinator(area)
    await coord.init()
    if coord.is_done:
        logger.info("Area done or queue empty: %s / %s", area.province, area.district)
        _write_done(area, coord)
        return

    global _http_server
    runtime_dir = os.path.dirname(area.map_viewer_file)
    if _http_server is None:
        _http_server = _start_map_server(runtime_dir, MAP_SERVER_PORT)
    map_url = f"http://127.0.0.1:{MAP_SERVER_PORT}/{os.path.basename(area.map_viewer_file)}"
    logger.info("Map viewer opened: %s", map_url)
    webbrowser.open(map_url)

    workers = [Worker(i, coord) for i in range(NUM_WORKERS)]
    tasks = [asyncio.create_task(w.run(pw)) for w in workers]
    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            exc = task.exception()
            if exc:
                logger.error("Worker crashed: %s", exc, exc_info=exc)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        remaining = [t for t in tasks if not t.done()]
        if remaining:
            await asyncio.gather(*remaining, return_exceptions=True)

    stats = coord.stats
    logger.info("  DONE area %s / %s", area.province, area.district)
    logger.info("  Tiles processed : %d / %d (%.1f%%)", stats["tiles_done"], stats["total_tiles"], stats["pct_done"])
    logger.info("  POIs collected  : %d", stats["pois_found"])
    logger.info("  Results saved   → %s", area.results_file)
    if coord.is_done:
        _write_done(area, coord)


def _write_done(area: ScanArea, coord: Coordinator) -> None:
    path = Path(area.runtime_dir) / "done.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"province": area.province, "district": area.district, "stats": coord.stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def main() -> None:
    _setup_logging()
    args = _parse_args()
    areas = _areas_from_args(args)
    logger.info("Google Maps POI Scraper — mode=%s workers=%d areas=%d", args.scan_mode, NUM_WORKERS, len(areas))
    logger.info("Viewport: %dx%d px", config.SCREENSHOT_W, config.SCREENSHOT_H)
    if not areas:
        logger.warning("No scan areas found. Download boundaries/manifest first.")
        return

    async with async_playwright() as pw:
        for area in areas:
            await _run_area(area, pw)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
