import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(f"Total POIs: {len(d)}")
for i, p in enumerate(d, 1):
    print(f"{i:2d}. {p['name']}")
