import asyncio
import logging
import os
import signal
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

# Đảm bảo thư mục hiện tại luôn nằm trong sys.path để tránh lỗi ModuleNotFoundError
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Windows terminal encoding (cp1252 không hỗ trợ tiếng Việt có dấu)
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

from config import LOG_FILE, LOG_LEVEL, NUM_WORKERS
from coordinator import Coordinator
from map_viewer import build_and_save as _build_map
from worker import Worker


# ── Logging setup ─────────────────────────────────────────────

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

MAP_SERVER_PORT = 8765  # HTTP server phục vụ map_viewer.html qua localhost
_http_server: ThreadingHTTPServer | None = None


def _start_map_server(directory: str, port: int) -> ThreadingHTTPServer:
    """Khởi chạy HTTP server nhỏ phục vụ map_viewer.html (cho phép fetch() hoạt động)."""
    class _Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        def log_message(self, *args):  # noqa: tắt log request thừa
            pass

    server = ThreadingHTTPServer(('127.0.0.1', port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# ── Main async ───────────────────────────────────────────────

async def main() -> None:
    _setup_logging()

    logger.info("=" * 60)
    logger.info("  OSM POI Scraper — Local AI Edition")
    logger.info("  Workers: %d", NUM_WORKERS)
    logger.info("=" * 60)

    # Khởi tạo coordinator
    coord = Coordinator()
    await coord.init()

    if coord.is_done:
        logger.info("Queue is empty — nothing to do. "
                    "Delete checkpoint.json to restart.")
        return

    # Khởi động HTTP server để phục vụ map_viewer.html qua localhost
    project_dir = os.path.dirname(os.path.abspath(__file__))
    global _http_server
    _http_server = _start_map_server(project_dir, MAP_SERVER_PORT)

    # Mở bản đồ trong trình duyệt (qua HTTP → fetch() hoạt động)
    map_path = _build_map(coord._all_tiles, coord._visited, coord._queued, coord._results, coord._discarded)
    map_url = f"http://127.0.0.1:{MAP_SERVER_PORT}/map_viewer.html"
    logger.info("Map viewer opened: %s", map_url)
    webbrowser.open(map_url)

    # Chạy workers dưới Playwright context
    async with async_playwright() as pw:
        workers = [Worker(i, coord) for i in range(NUM_WORKERS)]
        tasks = [asyncio.create_task(w.run(pw)) for w in workers]

        try:
            # Chờ hoàn thành
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_EXCEPTION,
            )
            # Nếu có exception
            for task in done:
                exc = task.exception()
                if exc:
                    logger.error("Worker crashed: %s", exc, exc_info=exc)
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt (Ctrl+C) detected — shutting down all workers gracefully…")
            raise
        finally:
            # Huỷ tasks còn lại
            for task in tasks:
                if not task.done():
                    task.cancel()
            remaining = [t for t in tasks if not t.done()]
            if remaining:
                await asyncio.gather(*remaining, return_exceptions=True)

    # In tóm tắt
    stats = coord.stats
    logger.info("=" * 60)
    logger.info("  DONE")
    logger.info("  Tiles processed : %d / %d (%.1f%%)",
                stats["tiles_done"], stats["total_tiles"], stats["pct_done"])
    logger.info("  POIs collected  : %d", stats["pois_found"])
    logger.info("  Queue remaining : %d", stats["queue_size"])
    logger.info("  Results saved   → results.json")
    logger.info("=" * 60)
    logger.info("Run `python clean_data.py` to deduplicate and export CSV.")


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
