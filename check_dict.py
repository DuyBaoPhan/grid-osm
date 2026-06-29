"""Kiểm tra và bổ sung osm_words.json với các từ còn thiếu."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('data/osm_words.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(f"Total entries: {len(d)}")

# Kiểm tra các từ cần cho Q1
check = [
    'buu dien', 'duong sach', 'bai giu', 'bai do', 'vuon', 'vuon hoa',
    'nha hang', 'quan an', 'quan ca phe', 'ca phe',
    'nguyen hue', 'dong khoi', 'le loi', 'hai ba trung',
    'dinh tien hoang', 'bao tang', 'benh vien', 'ngan hang',
    'truong', 'chua', 'nha tho', 'cho', 'sieu thi',
    'san bay', 'ben xe', 'cong vien',
    'pho', 'bun', 'com', 'giat ui',
    'tiem', 'cua hang', 'khach san',
]
print("\nDictionary lookup:")
for kw in check:
    val = d.get(kw)
    status = f"✓ → {val!r}" if val else "✗ not found"
    print(f"  {kw:25s} {status}")
