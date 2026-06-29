"""Test với các OCR output thực tế từ log."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.vietnam_places import normalize_place_phrases, _load_osm_words, _load_places
# Clear cache to reload fresh
_load_osm_words.cache_clear()
_load_places.cache_clear()

# Actual OCR outputs từ scraper.log
real_tests = [
    # Tile (0,0) - từ log thực tế
    "AO DAI AND AO BA BA RENTALS",
    "Buu dien trung tam Sai Gon",
    "Hum Central - Healthy Veggies Delights",
    "LPBank PGD Buu dien Giao dich Sai Gon",
    "Bai giu xe Duong Sach Highlands",
    "Highlands Coffee Saigon Post Office",
    "Cong Duong sach TP. Ho Chi Minh",
    "Vuon Trong Pho, Gia Dinh Connection",
    "Con Mao Nho Little Cats Studio",
    "PASTA CLUB Not so Italian",
    # Tile (0,-1)
    "Destiny Winter Clothes Store",
    "Capi Studio DIY Souvenirs 8",
    "The Box Market",
    "MCM Post Office",
]

print("=== Real OCR Test ===\n")
for t in real_tests:
    result = normalize_place_phrases(t)
    changed = " → " + result if result != t else " (unchanged)"
    print(f"  Input:  {t!r}")
    if result != t:
        print(f"  Result: {result!r}")
    print()
