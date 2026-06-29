import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('data/osm_words.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

# Problematic tokens
for k in ['con', 'cong', 'pho', 'phu', 'da', 'dong', 'giu', 'sua', 'bai']:
    if k in d:
        print(f"  {k!r} -> {d[k]!r}")
    else:
        print(f"  {k!r} not found")
