"""benchmark_ocr.py — Đánh giá độ chính xác OCR trực tiếp trên ảnh crop.

Tự động tạo bộ dữ liệu có nhãn (manifest) từ tên file ảnh crop, chạy qua pipeline
VietOCR + cleanup và so khớp để tính các chỉ số:
  - Exact Match (EM)
  - Token F1 Score
  - Tỷ lệ từ rác, sai lệch dấu

Sử dụng:
  python scripts/benchmark_ocr.py --crops crops
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vietnam_places import strip_vietnamese_accents
from src.vision import _clean_final_ocr_text, _get_vietocr_predictor, _recognize_text_crop_vietocr

# Mẫu tên file: tile_0_0_poi_4_Cổng_Đường_sách__TP._Hồ_Chí_Minh.png
# __ đại diện cho dấu gạch chéo phân cách phân đoạn (/)
_FILENAME_PATTERN = re.compile(r"^tile_-?\d+_-?\d+_poi_\d+_(.+)\.png$")


def parse_expected_name_from_filename(filename: str) -> str:
    match = _FILENAME_PATTERN.match(filename)
    if not match:
        return ""
    raw_part = match.group(1)
    # Thay __ bằng ' / ' để phục hồi các phân đoạn
    segments = raw_part.split("__")
    restored_segments = []
    for seg in segments:
        # Thay thế các gạch dưới đơn bằng dấu cách
        restored_segments.append(seg.replace("_", " ").strip())
    return " / ".join(restored_segments)


def calculate_token_f1(actual: str, expected: str) -> float:
    def get_tokens(text: str) -> list[str]:
        return re.findall(r"[A-Za-zÀ-ỹ0-9]+", strip_vietnamese_accents(text).lower())

    act_tokens = get_tokens(actual)
    exp_tokens = get_tokens(expected)
    if not act_tokens or not exp_tokens:
        return 0.0 if act_tokens != exp_tokens else 1.0

    overlap = 0
    remaining = exp_tokens.copy()
    for t in act_tokens:
        if t in remaining:
            remaining.remove(t)
            overlap += 1

    precision = overlap / len(act_tokens)
    recall = overlap / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_benchmark(crops_dir: str, manifest_path: str, update_manifest: bool = False):
    crops_path = Path(crops_dir)
    if not crops_path.exists():
        print(f"Lỗi: Thư mục ảnh crop '{crops_dir}' không tồn tại.")
        print("Hãy chạy 'py main.py' để thu thập một số ô và tự động sinh ảnh crop trước.")
        return

    # 1. Load hoặc tạo manifest
    manifest = {}
    if Path(manifest_path).exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            print(f"Đã load {len(manifest)} nhãn từ manifest {manifest_path}")
        except Exception as e:
            print(f"Cảnh báo: Không thể đọc manifest: {e}. Sẽ sinh tự động.")

    # Tìm tất cả file ảnh crop
    crop_files = sorted([f for f in os.listdir(crops_path) if f.endswith(".png")])
    if not crop_files:
        print(f"Không tìm thấy ảnh crop nào trong '{crops_dir}'.")
        return

    print(f"Tìm thấy {len(crop_files)} ảnh crop trong '{crops_dir}'.")

    # Tự động gán nhãn cho các file chưa có trong manifest
    manifest_updated = False
    for f in crop_files:
        if f not in manifest or update_manifest:
            expected = parse_expected_name_from_filename(f)
            if expected:
                manifest[f] = expected
                manifest_updated = True

    if manifest_updated:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"Đã cập nhật và lưu {len(manifest)} nhãn vào manifest {manifest_path}")

    # 2. Khởi động predictor VietOCR
    print("Đang tải mô hình VietOCR...")
    predictor = _get_vietocr_predictor()
    if predictor is None:
        print("Lỗi: Không thể tải mô hình VietOCR.")
        return

    # 3. Chạy benchmark
    results = []
    total_token_f1 = 0.0
    total_exact_match = 0

    print("\nBắt đầu chạy benchmark trên từng ảnh crop:")
    print("=" * 80)

    for idx, filename in enumerate(crop_files, 1):
        expected = manifest.get(filename)
        if not expected:
            continue

        img_path = crops_path / filename
        try:
            with open(img_path, "rb") as f:
                cv_img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            cv_img = None
        if cv_img is None:
            print(f"[{idx:3d}] ✗ Không thể đọc ảnh: {filename}")
            continue

        # Chạy OCR
        h, w = cv_img.shape[:2]
        # Bbox giả định bao quanh toàn bộ ảnh crop
        bbox = [0, 0, w, h]
        
        try:
            # Chạy hàm nhận diện chuẩn của vision.py
            raw_text = _recognize_text_crop_vietocr(cv_img, bbox, scale=1.0)
            actual = _clean_final_ocr_text(raw_text)
        except Exception as e:
            print(f"[{idx:3d}] ✗ Lỗi khi chạy OCR cho {filename}: {e}")
            continue

        f1 = calculate_token_f1(actual, expected)
        em = 1 if actual.strip().lower() == expected.strip().lower() else 0

        total_token_f1 += f1
        total_exact_match += em

        status_char = "✓" if em else ("~" if f1 >= 0.8 else "✗")
        print(f"[{idx:3d}] {status_char} File: {filename}")
        print(f"      Expected: {expected!r}")
        print(f"      Actual:   {actual!r} (F1: {f1:.2%})")

        results.append({
            "filename": filename,
            "expected": expected,
            "actual": actual,
            "em": em,
            "f1": f1
        })

    # 4. In báo cáo tổng hợp
    total_evaluated = len(results)
    if total_evaluated == 0:
        print("\nKhông có kết quả đánh giá nào.")
        return

    avg_f1 = total_token_f1 / total_evaluated
    em_rate = total_exact_match / total_evaluated

    print("=" * 80)
    print("KẾT QUẢ ĐÁNH GIÁ BENCHMARK")
    print("=" * 80)
    print(f"Tổng số ảnh đánh giá:  {total_evaluated}")
    print(f"Exact Match (EM):      {total_exact_match} ({em_rate:.2%})")
    print(f"Average Token F1:     {avg_f1:.2%}")
    print("=" * 80)

    # Xuất báo cáo HTML chi tiết
    html_path = crops_path.parent / "ocr_benchmark_report.html"
    generate_html_report(results, em_rate, avg_f1, str(html_path))
    print(f"Đã xuất báo cáo chi tiết dạng HTML tại: {html_path}")


def generate_html_report(results: list[dict], em_rate: float, avg_f1: float, output_path: str):
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>OCR Accuracy Benchmark Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f8f9fa; }}
        h1, h2 {{ color: #212529; }}
        .summary {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 25px; display: flex; gap: 40px; }}
        .metric {{ display: flex; flex-direction: column; }}
        .metric-val {{ font-size: 32px; font-weight: bold; color: #0d6efd; }}
        .metric-label {{ font-size: 14px; color: #6c757d; text-transform: uppercase; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background-color: #f1f3f5; color: #495057; font-weight: 600; }}
        tr:hover {{ background-color: #f8f9fa; }}
        .status-ok {{ background-color: #d1e7dd; color: #0f5132; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
        .status-warn {{ background-color: #fff3cd; color: #664d03; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
        .status-fail {{ background-color: #f8d7da; color: #842029; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
        img {{ max-height: 50px; border: 1px solid #ced4da; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Báo cáo đánh giá chất lượng OCR địa danh</h1>
    <div class="summary">
        <div class="metric">
            <span class="metric-val">{len(results)}</span>
            <span class="metric-label">Tổng số ảnh</span>
        </div>
        <div class="metric">
            <span class="metric-val">{em_rate:.2%}</span>
            <span class="metric-label">Exact Match (EM)</span>
        </div>
        <div class="metric">
            <span class="metric-val">{avg_f1:.2%}</span>
            <span class="metric-label">Token F1 Score</span>
        </div>
    </div>

    <h2>Bảng so khớp chi tiết</h2>
    <table>
        <thead>
            <tr>
                <th>STT</th>
                <th>Ảnh crop</th>
                <th>Tên file</th>
                <th>Nhãn đúng (Expected)</th>
                <th>OCR đọc được (Actual)</th>
                <th>Token F1</th>
                <th>Trạng thái</th>
            </tr>
        </thead>
        <tbody>
    """

    for idx, r in enumerate(results, 1):
        f1 = r["f1"]
        em = r["em"]
        if em:
            status = '<span class="status-ok">Khớp 100%</span>'
        elif f1 >= 0.8:
            status = '<span class="status-warn">Lệch nhẹ</span>'
        else:
            status = '<span class="status-fail">Sai/Mất chữ</span>'

        img_rel_path = f"crops/{r['filename']}"

        html_content += f"""
            <tr>
                <td>{idx}</td>
                <td><img src="{img_rel_path}" alt="crop" onerror="this.style.display='none'"></td>
                <td style="font-size: 12px; color: #6c757d;">{r['filename']}</td>
                <td style="font-weight: 500;">{r['expected']}</td>
                <td style="color: { '#198754' if em else '#dc3545' }; font-weight: 500;">{r['actual']}</td>
                <td style="font-weight: bold;">{f1:.1%}</td>
                <td>{status}</td>
            </tr>
        """

    html_content += """
        </tbody>
    </table>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Crop Accuracy Benchmarking")
    parser.add_argument("--crops", default="crops", help="Thư mục chứa ảnh crop POI")
    parser.add_argument("--manifest", default="data/ocr_benchmark_manifest.json", help="Đường dẫn file manifest JSON")
    parser.add_argument("--update-manifest", action="store_true", help="Ghi đè manifest bằng nhãn trích xuất từ tên file")
    args = parser.parse_args()

    run_benchmark(args.crops, args.manifest, args.update_manifest)
