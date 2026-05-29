import asyncio
import logging
import os
import signal
import sys
import webbrowser

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

    # Mo ban do trong trinh duyet
    map_path = _build_map(coord._all_tiles, coord._visited, coord._queued, len(coord._results), coord._discarded)
    map_url = "file:///" + map_path.replace(os.sep, "/")
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
