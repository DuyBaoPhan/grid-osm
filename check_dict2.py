"""Kiểm tra và sửa vấn đề trong OSM dictionary."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('data/osm_words.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

# 1. Tìm "so" entry
print("Looking for 'so':")
for k in ['so', 'no', 'to', 'not', 'of', 'and', 'the', 'in', 'at', 'on']:
    if k in d:
        print(f"  PROBLEM: {k!r} -> {d[k]!r}")

# 2. Tìm "trung" entry
print("\nLooking for 'trung':")
for k in ['trung', 'trong', 'truong', 'trang', 'tran']:
    if k in d:
        print(f"  {k!r} -> {d[k]!r}")
