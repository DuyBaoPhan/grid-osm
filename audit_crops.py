import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
CROPS = ROOT / "crops"
RESULTS = ROOT / "results.json"
OUT_HTML = ROOT / "audit_crops.html"
OUT_JSON = ROOT / "audit_report.json"


def crop_text_from_name(name: str) -> str:
    s = re.sub(r"^tile_-?\d+_-?\d+_poi_\d+_", "", name)
    s = re.sub(r"\.png$", "", s, flags=re.I)
    s = s.replace("__", " / ").replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def norm(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("/", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9à-ỹđ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def main():
    results = []
    if RESULTS.exists():
        with open(RESULTS, "r", encoding="utf-8") as f:
            results = json.load(f)
    result_names = [r.get("name", "") for r in results]
    crops = sorted(CROPS.glob("*.png"), key=lambda p: p.stat().st_mtime)

    rows = []
    report = []
    for crop in crops:
        ctext = crop_text_from_name(crop.name)
        best_idx = None
        best_score = -1.0
        best_name = ""
        for i, rname in enumerate(result_names):
            score = sim(ctext, rname)
            if score > best_score:
                best_score = score
                best_idx = i
                best_name = rname
        status = "OK" if best_score >= 0.92 else ("REVIEW" if best_score >= 0.75 else "MISSING")
        report.append({
            "crop": crop.name,
            "crop_text": ctext,
            "best_result_index": best_idx,
            "best_result_name": best_name,
            "score": round(best_score, 4),
            "status": status,
        })
        badge = {"OK": "✅", "REVIEW": "⚠️", "MISSING": "❌"}[status]
        rows.append(f"""
        <tr class="{status.lower()}">
          <td>{len(report)}</td>
          <td><img src="crops/{crop.name}" loading="lazy"></td>
          <td class="text">{ctext}</td>
          <td class="text">{best_name}</td>
          <td>{best_score:.3f}</td>
          <td>{badge} {status}</td>
        </tr>""")

    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Crop OCR Audit</title>
<style>
body{{font-family:Arial,sans-serif;background:#111;color:#eee;margin:20px}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #333;padding:8px;vertical-align:top}}
img{{max-width:420px;max-height:220px;background:#fff}}
.text{{font-size:18px;line-height:1.35}}
th{{position:sticky;top:0;background:#222}}
.ok{{background:#102015}} .review{{background:#2a250d}} .missing{{background:#2a1010}}
</style></head><body>
<h1>Crop OCR Audit</h1>
<p>Crops: {len(crops)} | Results: {len(results)}</p>
<table><thead><tr><th>#</th><th>Crop image</th><th>Crop filename text</th><th>Best result name</th><th>Score</th><th>Status</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")
    counts = {"OK": 0, "REVIEW": 0, "MISSING": 0}
    for item in report:
        counts[item["status"]] += 1
    print(f"Crops={len(crops)} Results={len(results)} OK={counts['OK']} REVIEW={counts['REVIEW']} MISSING={counts['MISSING']}")
    print(f"Wrote {OUT_HTML}")
    print(f"Wrote {OUT_JSON}")
    for item in report:
        if item["status"] != "OK":
            print(f"{item['status']}: score={item['score']} crop={item['crop_text']} | result={item['best_result_name']}")

if __name__ == "__main__":
    main()
