import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
with open('data/osm_words.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
for k in ['gia', 'gia dinh', 'dinh', 'mao', 'nho']:
    v = d.get(k)
    val = repr(v) if v else 'not found'
    print(f"  {k!r} -> {val}")
