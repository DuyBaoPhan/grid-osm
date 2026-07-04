# =============================================================
# recognizers.py — OCR recognizers extracted from original vision.py
# =============================================================

import io
import logging
import re
from typing import List

import cv2
import numpy as np
from PIL import Image

from .models import _get_paddle_text_detector, _get_vietocr_predictor
from .geometry import detect_text_area
from .text_cleaning import (
    _append_missing_known_suffix,
    _clean_final_ocr_text,
    _clean_junk_words,
    _clean_ocr_edge_segments,
    _clean_spelling,
    _contains_token_subsequence,
    _is_clean_short_brand_candidate,
    _is_junk_line,
    _is_known_token,
    _junk_token_count,
    _looks_like_bad_ocr,
    _looks_like_junk_token,
    _looks_like_vietnamese_gibberish,
    _merge_best_diacritics,
    _merge_missing_middle_tokens,
    _merge_overlapping_ocr_continuation,
    _merge_primary_diacritics,
    _normalized_adds_suspicious_text,
    _normalized_has_valid_main_name_extension,
    _normalized_regresses_quality,
    _ocr_artifact_score,
    _ocr_tokens,
    _remove_adjacent_duplicate_ocr_tokens,
    _score_ocr_text_quality,
    _strip_vietnamese_accents,
    _texts_are_unrelated,
)
from .crop_processing import (
    _select_primary_line_crops,
    normalize_ocr_background,
    split_crop_into_lines,
    split_crop_into_lines_normalized,
)

logger = logging.getLogger(__name__)

def _recognize_text_paddle(cv_img: np.ndarray) -> str:
    """Fallback recognition bằng PaddleOCR trên crop đã chọn."""
    detector = _get_paddle_text_detector()
    if detector is None or cv_img is None or cv_img.size == 0:
        return ""
    try:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = detector.predict(rgb)
        texts = []
        for res in results if isinstance(results, list) else [results]:
            if isinstance(res, dict):
                for key in ("rec_texts", "texts"):
                    vals = res.get(key)
                    if isinstance(vals, list):
                        texts.extend(str(v).strip() for v in vals if str(v).strip())
                if "rec_text" in res and str(res["rec_text"]).strip():
                    texts.append(str(res["rec_text"]).strip())
            elif isinstance(res, (list, tuple)):
                for item in res:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        rec = item[1]
                        if isinstance(rec, (list, tuple)) and rec:
                            texts.append(str(rec[0]).strip())
                        elif isinstance(rec, str):
                            texts.append(rec.strip())
        cleaned = []
        for t in texts:
            t = _clean_junk_words(_clean_spelling(t))
            if t and not _is_junk_line(t):
                cleaned.append(t)
        return " / ".join(cleaned)
    except Exception as exc:
        logger.debug("PaddleOCR recognition fallback error: %s", exc)
        return ""


def _recognize_text_crop_vietocr_normalized(
    cv_img: np.ndarray,
    bbox: List[float],
    icon_side: str = "left",
    scale: float = 1.0,
    cx: float = None,
    cy: float = None
) -> str:
    """Hàm OCR phụ chạy trên nền được chuẩn hóa hoàn toàn và xử lý icon triệt để."""
    predictor = _get_vietocr_predictor()
    if predictor is None or cv_img is None:
        return ""

    try:
        h_img, w_img = cv_img.shape[:2]
        x1_orig, y1_orig, x2_orig, y2_orig = map(int, bbox)
        
        pad = int(16 * scale)
        y1 = max(0, y1_orig - pad)
        y2 = min(h_img, y2_orig + pad)
        x1 = max(0, x1_orig - pad)
        x2 = min(w_img, x2_orig + pad)
        
        raw_crop = cv_img[y1:y2, x1:x2]
        if raw_crop.size == 0:
            return ""

        # 1. Khử nền bằng logic cải tiến
        base_norm_img = normalize_ocr_background(raw_crop)
        h_rc, w_rc = base_norm_img.shape[:2]

        bg_color = [245, 242, 232]  # BGR ngà nhạt giống crop đọc tốt
        cx_local = (cx - x1) if cx is not None else None
        cy_local = (cy - y1) if cy is not None else None

        def _run_normalized_variant(mask_icon: bool) -> str:
            """Chạy OCR normalized với/không mask icon để tránh cắt mất chữ đầu."""
            norm_img = base_norm_img.copy()

            if mask_icon and cx_local is not None and cy_local is not None:
                if icon_side == "left":
                    mask_w = max(0, min(w_rc, int(cx_local + 12.0 * scale)))
                    norm_img[:, 0:mask_w] = bg_color
                elif icon_side == "right":
                    mask_x = max(0, min(w_rc, int(cx_local - 18.0 * scale)))
                    norm_img[:, mask_x:w_rc] = bg_color
                elif icon_side == "top":
                    mask_h = max(0, min(h_rc, int(cy_local + 18.0 * scale)))
                    norm_img[0:mask_h, :] = bg_color

            tx1, ty1, tx2, ty2 = detect_text_area(norm_img, scale)
            tx1_safe = max(0, tx1 - int(2 * scale))
            tx2_safe = min(w_rc, tx2 + int(4 * scale))
            crop = norm_img[ty1:ty2, tx1_safe:tx2_safe]
            if crop.size == 0:
                return ""

            line_crops = _select_primary_line_crops(split_crop_into_lines_normalized(crop, scale))
            variant_texts = []
            for line_crop in line_crops:
                if line_crop.size == 0:
                    continue
                line_rgb = cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB)
                bg_color_line = line_crop[0, 0].tolist()
                pad_h = max(4, int(6 * scale))
                pad_w = max(8, int(12 * scale))
                padded_line = cv2.copyMakeBorder(
                    line_rgb,
                    pad_h, pad_h, pad_w, pad_w,
                    cv2.BORDER_CONSTANT,
                    value=bg_color_line
                )

                def _clean_ocr_line(text: str) -> str:
                    # Strip any rating-like prefix ending with a number in parentheses, e.g. "4.14 (382) - " or "J4x(382) - "
                    text_clean = re.sub(r'^.*?\(\d+(?:[.,]\d+)?\s*[KkM]?[+-]?\)\s*(?:[-·•*]\s*)?', '', (text or '').strip()).strip()
                    text_clean = _clean_junk_words(text_clean)
                    if text_clean and not _is_junk_line(text_clean):
                        return text_clean
                    return ""

                def _line_score(text: str) -> int:
                    return _score_ocr_text_quality(text)

                candidates = []
                for fx in (1, 2, 3):
                    if fx == 1:
                        candidate_img = padded_line
                    else:
                        candidate_img = cv2.resize(
                            padded_line,
                            None,
                            fx=fx,
                            fy=fx,
                            interpolation=cv2.INTER_CUBIC
                        )
                    raw_text = (predictor.predict(Image.fromarray(candidate_img)) or "").strip()
                    clean_text = _clean_ocr_line(raw_text)
                    if clean_text:
                        score = _score_ocr_text_quality(clean_text, fx)
                        candidates.append((score, clean_text))

                if candidates:
                    text_clean = max(candidates, key=lambda x: x[0])[1]
                    variant_texts.append(text_clean)

            return " / ".join(variant_texts)

        masked_text = _run_normalized_variant(mask_icon=True)
        unmasked_text = _run_normalized_variant(mask_icon=False)

        def _variant_score(text: str) -> int:
            clean_len = len(re.sub(r'[^A-Za-zÀ-ỹ0-9]', '', text or ""))
            line_bonus = 12 * max(0, (text or "").count(" / "))
            bad_penalty = 80 if _looks_like_bad_ocr(text) else 0
            return clean_len + line_bonus - bad_penalty

        # Nếu crop đã sạch icon, bản không mask thường giữ được chữ đầu (Little, sách, Mặn...).
        # Chỉ chọn unmasked khi nó tốt hơn rõ ràng, tránh ảnh hưởng các POI có icon thật.
        if unmasked_text and _variant_score(unmasked_text) >= _variant_score(masked_text) + 5:
            return unmasked_text

        return masked_text
    except Exception as exc:
        logger.debug("Lỗi trong recognize_text_crop_vietocr_normalized: %s", exc)
        return ""


def _recognize_text_crop_vietocr(cv_img: np.ndarray, bbox: List[float], icon_side: str = "left", scale: float = 1.0, cx: float = None, cy: float = None) -> str:
    predictor = _get_vietocr_predictor()
    if predictor is None or cv_img is None:
        return ""

    h_img, w_img = cv_img.shape[:2]
    x1_orig, y1_orig, x2_orig, y2_orig = map(int, bbox)
    
    pad = int(10 * scale)
    y1 = max(0, y1_orig - pad)
    y2 = min(h_img, y2_orig + pad)
    x1 = max(0, x1_orig - pad)
    x2 = min(w_img, x2_orig + pad)
    
    raw_crop = cv_img[y1:y2, x1:x2]
    if raw_crop.size == 0:
        return ""

    # Tiền xử lý làm sạch nền:
    # 1. Tính màu nền chủ đạo bằng median BGR của raw_crop
    bg_color = np.median(raw_crop, axis=(0, 1)).astype(int).tolist()

    # 2. Tạo bản sao sạch và tô đè màu nền lên vùng padding 10px ngoài
    crop_clean = raw_crop.copy()
    h_rc, w_rc = crop_clean.shape[:2]
    border_w = int(10 * scale)
    if border_w > 0:
        if border_w < h_rc:
            crop_clean[0:border_w, :] = bg_color
            crop_clean[h_rc - border_w:, :] = bg_color
        if border_w < w_rc:
            crop_clean[:, 0:border_w] = bg_color
            crop_clean[:, w_rc - border_w:] = bg_color

    def _ocr_from_prepared_crop(prepared_crop: np.ndarray) -> str:
        tx1, ty1, tx2, ty2 = detect_text_area(prepared_crop, scale)
        # Nới nhẹ mép trái để tránh mất nét dọc đầu chữ như H/L/T khi crop sát icon.
        tx1_safe = max(0, tx1 - int(6 * scale))
        tx2_safe = min(prepared_crop.shape[1], tx2 + int(3 * scale))
        crop_local = prepared_crop[ty1:ty2, tx1_safe:tx2_safe]

        if crop_local.size <= 0:
            return ""

        try:
            line_crops = _select_primary_line_crops(split_crop_into_lines(crop_local, scale))
            
            texts = []

            def _best_text_for_line(line_img: np.ndarray) -> str:
                if line_img is None or line_img.size == 0:
                    return ""
                bg_color_line = line_img[0, 0].tolist()
                pad_h = max(4, int(6 * scale))
                pad_w = max(10, int(16 * scale))
                padded_line = cv2.copyMakeBorder(
                    line_img,
                    pad_h, pad_h, pad_w, pad_w,
                    cv2.BORDER_CONSTANT,
                    value=bg_color_line
                )
                rgb_line = cv2.cvtColor(padded_line, cv2.COLOR_BGR2RGB)
                candidates = []
                for fx in (1, 2, 3):
                    im = rgb_line if fx == 1 else cv2.resize(rgb_line, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
                    raw_text = (predictor.predict(Image.fromarray(im)) or "").strip()
                    text_clean = re.sub(r'^.*?\(\d+(?:[.,]\d+)?\s*[KkM]?[+-]?\)\s*(?:[-·•*]\s*)?', '', raw_text).strip()
                    text_clean = _clean_junk_words(text_clean)
                    if text_clean:
                        score = _score_ocr_text_quality(text_clean, fx)
                        if score > -150:
                            candidates.append((score, text_clean))
                return max(candidates, key=lambda x: x[0])[1] if candidates else ""

            def _prepend_prefix_if_shared(base_text: str, alt_text: str) -> str:
                base_clean = _clean_final_ocr_text(base_text)
                alt_clean = _clean_final_ocr_text(alt_text)
                if not base_clean or not alt_clean or base_clean == alt_clean:
                    return base_clean
                base_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', base_clean)
                alt_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', alt_clean)
                base_keys = [_strip_vietnamese_accents(w).lower() for w in base_words]
                alt_keys = [_strip_vietnamese_accents(w).lower() for w in alt_words]
                for i in range(1, min(4, len(alt_keys)) + 1):
                    prefix_words = alt_words[:i]
                    if len(prefix_words) > 3:
                        break
                    if _is_junk_line(" ".join(prefix_words)) or _junk_token_count(" ".join(prefix_words)):
                        continue
                    prefix_keys = alt_keys[:i]
                    max_overlap = min(len(prefix_keys), len(base_keys))
                    for overlap in range(max_overlap, 0, -1):
                        if prefix_keys[-overlap:] != base_keys[:overlap]:
                            continue
                        merged_words = prefix_words[:-overlap] + base_words
                        if not merged_words or merged_words == base_words:
                            continue
                        merged = " ".join(merged_words)
                        merged_clean = _clean_final_ocr_text(merged)
                        if merged_clean[:1].islower() and base_clean[:1].isupper():
                            merged_clean = merged_clean[:1].upper() + merged_clean[1:]
                        if _score_ocr_text_quality(merged_clean) >= _score_ocr_text_quality(base_clean) - 12:
                            return merged_clean
                    for j in range(0, len(base_keys)):
                        shared = 0
                        while i + shared < len(alt_keys) and j + shared < len(base_keys) and alt_keys[i + shared] == base_keys[j + shared]:
                            shared += 1
                        if shared >= 2 or (
                            shared >= 1
                            and j == 0
                            and prefix_words
                            and all(len(w) >= 3 for w in prefix_words)
                            and _score_ocr_text_quality(" ".join(prefix_words + base_words)) >= _score_ocr_text_quality(base_clean) - 12
                        ):
                            # If prefix probe overlaps the base at boundary, keep only new leading tokens.
                            if prefix_keys[-1] == base_keys[j]:
                                merged_words = prefix_words[:-1] + base_words
                            else:
                                merged_words = prefix_words + base_words
                            if not merged_words or merged_words == base_words:
                                continue
                            merged = " ".join(merged_words)
                            merged_clean = _clean_final_ocr_text(merged)
                            if merged_clean[:1].islower() and base_clean[:1].isupper():
                                merged_clean = merged_clean[:1].upper() + merged_clean[1:]
                            if _score_ocr_text_quality(merged_clean) >= _score_ocr_text_quality(base_clean) - 12:
                                return merged_clean
                return base_clean

            for line_crop in line_crops:
                if line_crop.size == 0:
                    continue
                best_line_text = _best_text_for_line(line_crop)
                if best_line_text:
                    h_line, w_line = line_crop.shape[:2]
                    # Probe left side: full-line OCR can ignore a visible leading brand when line is wide.
                    if w_line >= int(90 * scale):
                        for frac in (0.45, 0.50, 0.58, 0.68):
                            left_w = min(w_line, max(int(70 * scale), int(w_line * frac)))
                            left_text = _best_text_for_line(line_crop[:, :left_w])
                            rescued = _prepend_prefix_if_shared(best_line_text, left_text)
                            if rescued != _clean_final_ocr_text(best_line_text):
                                best_line_text = rescued
                                break
                    if best_line_text and not _is_junk_line(best_line_text):
                        texts.append(best_line_text)
            return " / ".join(texts)
        except Exception as e:
            logger.debug("Primary VietOCR error: %s", e)
            return ""

    # 3. Vẽ đè vòng tròn màu nền lên vùng icon nếu có tọa độ cx, cy.
    # Chạy thêm bản không mask để tránh ăn mất chữ đầu khi bbox/icon sát chữ.
    crop_masked = crop_clean.copy()
    if cx is not None and cy is not None:
        cx_local = cx - x1
        cy_local = cy - y1
        if icon_side == "left":
            mask_w = max(0, min(w_rc, int(cx_local + 12.0 * scale)))
            crop_masked[:, 0:mask_w] = bg_color
        elif icon_side == "right":
            mask_x = max(0, min(w_rc, int(cx_local - 18.0 * scale)))
            crop_masked[:, mask_x:w_rc] = bg_color
        elif icon_side == "top":
            mask_h = max(0, min(h_rc, int(cy_local + 18.0 * scale)))
            crop_masked[0:mask_h, :] = bg_color
        else:
            r = int(16 * scale)
            cv2.circle(crop_masked, (int(cx_local), int(cy_local)), r, bg_color, -1)

    primary_masked = _ocr_from_prepared_crop(crop_masked)
    primary_unmasked = _ocr_from_prepared_crop(crop_clean)

    def _should_use_unmasked(masked_text: str, unmasked_text: str) -> bool:
        if not masked_text or not unmasked_text:
            return bool(unmasked_text and not masked_text)
        if _ocr_artifact_score(unmasked_text) > _ocr_artifact_score(masked_text) + 2:
            return False
        if re.search(r'[)\]}>"“”]', unmasked_text) and not re.search(r'[)\]}>"“”]', masked_text):
            return False
        if _looks_like_bad_ocr(unmasked_text):
            return False

        masked_tokens = _ocr_tokens(masked_text)
        unmasked_tokens = _ocr_tokens(unmasked_text)
        if not masked_tokens or not unmasked_tokens:
            return False

        if (
            len(unmasked_tokens) > len(masked_tokens)
            and _contains_token_subsequence(unmasked_tokens, masked_tokens)
            and _score_ocr_text_quality(unmasked_text) >= _score_ocr_text_quality(masked_text) + 6
            and _ocr_artifact_score(unmasked_text) <= _ocr_artifact_score(masked_text) + 1
            and not _looks_like_bad_ocr(unmasked_text)
        ):
            return True

        m_known = sum(1 for t in masked_tokens if _is_known_token(t))
        u_known = sum(1 for t in unmasked_tokens if _is_known_token(t))

        if u_known > m_known and len(unmasked_tokens) <= len(masked_tokens) + 1:
            if _ocr_artifact_score(unmasked_text) <= _ocr_artifact_score(masked_text) + 1:
                return True

        # Nếu cùng số token, chỉ cho unmasked thắng khi nó thật sự khôi phục token bị cắt đầu.
        # Ví dụ cần cứu: tybrid -> hybrid (unmasked dài hơn 1 ký tự và chứa masked làm suffix).
        # Ví dụ phải giữ: tam -> vi tam (unmasked chèn token nhiễu), sai gon -> sai gon giữ nguyên.
        if len(masked_tokens) == len(unmasked_tokens):
            masked_raw_tokens = re.findall(r'[A-Za-zÀ-ỹ0-9]+', masked_text)
            improved_prefix = False
            worsened = False
            for idx, (m_tok, u_tok) in enumerate(zip(masked_tokens, unmasked_tokens)):
                if m_tok == u_tok:
                    continue
                raw_m = masked_raw_tokens[idx] if idx < len(masked_raw_tokens) else ""
                # Cứu cả token viết hoa nếu unmasked token là từ có trong từ điển
                if (raw_m[:1].islower() or _is_known_token(u_tok)) and len(u_tok) == len(m_tok) + 1 and u_tok.endswith(m_tok):
                    improved_prefix = True
                    continue
                worsened = True
                break
            return improved_prefix and not worsened

        # Nếu unmasked thêm token, thường là nhiễu icon/chữ lân cận.
        # Nhưng nếu masked nằm nguyên trong unmasked và phần thêm ở đầu có tín hiệu tên rõ,
        # thì masked đã cắt mất chữ đầu do mask icon quá rộng.
        if len(unmasked_tokens) > len(masked_tokens):
            if _contains_token_subsequence(unmasked_tokens, masked_tokens):
                for start in range(0, len(unmasked_tokens) - len(masked_tokens) + 1):
                    if unmasked_tokens[start:start + len(masked_tokens)] == masked_tokens:
                        leading = unmasked_tokens[:start]
                        trailing = unmasked_tokens[start + len(masked_tokens):]
                        if leading and not trailing:
                            unmasked_artifacts = _ocr_artifact_score(unmasked_text)
                            masked_artifacts = _ocr_artifact_score(masked_text)
                            if unmasked_artifacts <= masked_artifacts + 1 and not _looks_like_bad_ocr(unmasked_text):
                                return True
                        break
            return _looks_like_bad_ocr(masked_text)

        # Nếu unmasked ít token hơn, chỉ chọn khi masked có junk thật.
        # Không chọn chỉ vì subset: sẽ làm mất dòng hợp lệ như `đai - Chi nhánh Quận 1`.
        if len(unmasked_tokens) < len(masked_tokens):
            has_junk = any(_looks_like_junk_token(w) for w in masked_tokens) or (m_known < u_known)
            coverage = len(unmasked_tokens) / max(1, len(masked_tokens))
            if has_junk and coverage >= 0.55:
                return True

        return False

    if _should_use_unmasked(primary_masked, primary_unmasked):
        primary_text = primary_unmasked
        logger.info("  [OCR primary unmasked chosen] Masked='%s' -> Unmasked='%s'", primary_masked, primary_unmasked)
    else:
        primary_text = primary_masked

    # 4. Chạy luồng normalized OCR làm fallback
    norm_text = _recognize_text_crop_vietocr_normalized(cv_img, bbox, icon_side, scale, cx, cy)
    
    # 5. So sánh chất lượng và chọn kết quả tốt nhất bằng score tổng quát, không keyword.
    def _candidate_score(text: str, source: str) -> float:
        cleaned = _clean_final_ocr_text(text)
        if not cleaned:
            return -9999
        score = _score_ocr_text_quality(cleaned)
        # Không cộng bonus riêng cho normalized: log thực tế cho thấy normalized hay dài/sạch hơn nhưng đổi sai chữ gốc
        # như `Bưu` -> `Bir`, `Cổng` -> `Cống`, `19th-century` -> `9th-censury`.
        return score

    candidates = []
    for source, raw_text in (("primary", primary_text), ("normalized", norm_text)):
        cleaned = _clean_final_ocr_text(raw_text)
        if cleaned:
            candidates.append((_candidate_score(raw_text, source), source, cleaned, raw_text))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_source, selected_text, selected_raw = candidates[0]
        primary_candidate = next((item for item in candidates if item[1] == "primary"), None)
        if (
            best_source == "normalized"
            and primary_candidate
            and _is_clean_short_brand_candidate(primary_text)
            and _texts_are_unrelated(primary_text, norm_text)
        ):
            _, _, primary_cleaned, _ = primary_candidate
            selected_text = primary_cleaned
            best_source = "primary"
            logger.info(
                "  [OCR keep primary] Brand primary='%s' rejected unrelated Normalized='%s'",
                primary_cleaned,
                norm_text,
            )
        if (
            best_source == "normalized"
            and primary_candidate
            and not _looks_like_bad_ocr(primary_text)
            and _normalized_adds_suspicious_text(primary_text, norm_text)
            and not _normalized_has_valid_main_name_extension(primary_text, norm_text)
        ):
            _, _, primary_cleaned, _ = primary_candidate
            selected_text = primary_cleaned
            best_source = "primary"
            logger.info(
                "  [OCR keep primary] Primary='%s' rejected suspicious-extra Normalized='%s'",
                primary_cleaned,
                norm_text,
            )
        if best_source == "normalized" and primary_candidate and not _looks_like_bad_ocr(primary_text):
            primary_score, _, primary_cleaned, _ = primary_candidate
            norm_gain = best_score - primary_score
            if norm_gain < 9:
                primary_tokens = _ocr_tokens(primary_cleaned)
                norm_tokens = _ocr_tokens(selected_text)
                if (
                    len(primary_tokens) > len(norm_tokens)
                    and _contains_token_subsequence(primary_tokens, norm_tokens)
                    and _ocr_artifact_score(primary_cleaned) <= _ocr_artifact_score(selected_text) + 6
                ):
                    selected_text = primary_cleaned
                    best_source = "primary"
                    logger.info(
                        "  [OCR keep primary] Primary='%s' rejected Normalized='%s' dropped leading tokens (gain=%.1f)",
                        primary_cleaned,
                        selected_raw,
                        norm_gain,
                    )
                else:
                    m_known = sum(1 for t in _ocr_tokens(primary_text) if _is_known_token(t))
                    n_known = sum(1 for t in _ocr_tokens(selected_raw) if _is_known_token(t))
                    if n_known > m_known:
                        # Normalized has more known/valid tokens, keep it
                        pass
                    else:
                        selected_text = primary_cleaned
                        best_source = "primary"
                        logger.info(
                            "  [OCR keep primary] Primary='%s' rejected non-clear Normalized='%s' (gain=%.1f)",
                            primary_cleaned,
                            selected_raw,
                            norm_gain,
                        )
        is_normalized_chosen = best_source == "normalized"
        if best_source == "primary" and norm_text and _normalized_regresses_quality(primary_text, norm_text):
            logger.info("  [OCR keep primary] Primary='%s' rejected lower-quality Normalized='%s'", primary_text, norm_text)
        elif best_source == "normalized":
            logger.info("  [OCR normalized chosen] Primary='%s' -> Normalized='%s'", primary_text, selected_text)
    elif norm_text:
        selected_text = _clean_final_ocr_text(norm_text)
        is_normalized_chosen = True
    else:
        selected_text = _clean_final_ocr_text(primary_text)
        is_normalized_chosen = False

    if selected_text and primary_unmasked:
        unmasked_clean = _clean_final_ocr_text(primary_unmasked)
        selected_tokens = _ocr_tokens(selected_text)
        unmasked_tokens = _ocr_tokens(unmasked_clean)
        if (
            selected_tokens
            and len(unmasked_tokens) > len(selected_tokens)
            and _contains_token_subsequence(unmasked_tokens, selected_tokens)
            and _score_ocr_text_quality(unmasked_clean) >= _score_ocr_text_quality(selected_text) + 6
            and _ocr_artifact_score(unmasked_clean) <= _ocr_artifact_score(selected_text) + 1
            and not _looks_like_bad_ocr(unmasked_clean)
        ):
            logger.info("  [OCR prefix rescue] Selected='%s' + Unmasked='%s'", selected_text, unmasked_clean)
            selected_text = unmasked_clean

    if selected_text and primary_text and selected_text != primary_text:
        merged_text = _merge_best_diacritics(primary_text, selected_text)
        merged_text = _clean_final_ocr_text(merged_text)
        if merged_text and merged_text != selected_text:
            logger.info("  [OCR merge diacritics] Selected='%s' + Primary='%s' -> '%s'", selected_text, primary_text, merged_text)
            selected_text = merged_text

    for alt_source, alt_text in (("masked", primary_masked), ("unmasked", primary_unmasked), ("normalized", norm_text)):
        rescued_text = _append_missing_known_suffix(selected_text, alt_text)
        if rescued_text != selected_text:
            logger.info("  [OCR suffix rescue] Selected='%s' + %s='%s' -> '%s'", selected_text, alt_source, alt_text, rescued_text)
            selected_text = rescued_text
            break

    for alt_source, alt_text in (("masked", primary_masked), ("unmasked", primary_unmasked), ("normalized", norm_text)):
        rescued_text = _merge_missing_middle_tokens(selected_text, alt_text)
        if rescued_text != selected_text:
            logger.info("  [OCR missing-middle rescue] Selected='%s' + %s='%s' -> '%s'", selected_text, alt_source, alt_text, rescued_text)
            selected_text = rescued_text
            break

    for alt_source, alt_text in (("masked", primary_masked), ("unmasked", primary_unmasked), ("normalized", norm_text)):
        merged_text = _merge_overlapping_ocr_continuation(selected_text, alt_text)
        if merged_text != selected_text:
            logger.info("  [OCR overlap rescue] Selected='%s' + %s='%s' -> '%s'", selected_text, alt_source, alt_text, merged_text)
            selected_text = merged_text
            break
        
    # 6. Fallback sang PaddleOCR nếu cả hai luồng đều lỗi/rác
    if (
        not selected_text
        or _looks_like_bad_ocr(selected_text)
        or _looks_like_vietnamese_gibberish(selected_text)
        or _junk_token_count(selected_text) > 0
    ):
        tx1, ty1, tx2, ty2 = detect_text_area(crop_masked, scale)
        crop_for_paddle = crop_masked[ty1:ty2, max(0, tx1 - int(6 * scale)):tx2]
        paddle_text = _clean_final_ocr_text(_recognize_text_paddle(crop_for_paddle)) if crop_for_paddle.size > 0 else ""
        if paddle_text and not _looks_like_bad_ocr(paddle_text) and _junk_token_count(paddle_text) == 0:
            logger.info("  [OCR fallback] VietOCR selected='%s' -> PaddleOCR='%s'", selected_text, paddle_text)
            return paddle_text

    selected_text = _remove_adjacent_duplicate_ocr_tokens(_clean_final_ocr_text(selected_text))
    return selected_text
