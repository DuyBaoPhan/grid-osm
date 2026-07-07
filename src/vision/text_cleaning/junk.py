# Junk/category filtering helpers extracted from text_cleaning.py.

import re

from .dictionary import _is_known_token, _strip_vietnamese_accents

def _is_junk_line(s: str) -> bool:
    """Loại bỏ các dòng chữ rác không phải là tên địa điểm (số điện thoại, mã số thuế, đánh giá...)."""
    s = s.strip()
    if not s:
        return True

    # 1. Định dạng rating dạng "4.74 (53)" hoặc "(53)" hoặc "4.7"
    if re.match(r'^\d+(\.\d+)?\s*\(\d+\)$', s) or re.match(r'^\(\d+\)$', s):
        return True
    if re.match(r'^\d+(\.\d+)?$', s):
        return True

    # Mới: Loại bỏ các dòng rating kèm chữ phụ như "4.6% (35) - Đặt bàn ngay tại Hum"
    if re.search(r'^\d+(\.\d+)?%\s*\(\d+\)', s) or re.search(r'^\d+(\.\d+)?\s*\(\d+\)\s*-\s*', s):
        return True

    # Check for consecutive digits on the original string (with spaces/punctuation)
    # - 5 consecutive digits anywhere (e.g. "00110", "18000000...")
    # - or 3 consecutive zeros at the start (e.g. "000Trang...")
    if re.search(r'\d{5,}', s) or re.match(r'^0{3,}', s):
        return True

    # Remove spaces and common punctuation to analyze characters
    clean_s = re.sub(r'[^a-zA-Z0-9]', '', s)
    if not clean_s:
        return True

    digits = sum(c.isdigit() for c in clean_s)
    letters = sum(c.isalpha() for c in clean_s)
    total = len(clean_s)

    # 2. Định dạng chuỗi số dài (mã số thuế, số điện thoại, zip code...)
    # Nếu dòng chứa nhiều chữ số (trên 65% ký tự và có ít nhất 4 số)
    # Loại trừ trường hợp địa chỉ (ví dụ: "123 Lê Lợi" có chữ cái chiếm đa số)
    if total > 0 and digits >= 4 and (digits / total) > 0.65:
        if letters == 0 or (letters / total) < 0.20:
            return True

    # 3. Các từ viết tắt nhiễu do quét nhầm viền icon Google Maps (ví dụ: Cons, COLInters, commes...)
    # Nhận diện các từ rác bắt đầu bằng tiền tố và kết thúc bằng hậu tố đặc trưng của nét tròn
    junk_pattern = (
        r'(?i)^(con|col|com|comm|cor|cot|cos|can|trn|trans|contra|contr|inter|hype|como|trns|colinter|ant|antr|dis|disc|discol)'
        r'(s|es|ers|ins|inters|ess|he|ste|cars|c|ar|shone|gers|tracess|ces|cess|monums|phousness|anshone|cars|ication|che|tess|ness|mogers'
        r'|anaTis|naTis|interpLat|interpOUm|prOUm|OUm|com|intersm|intersM|vers|sstering|teris|tracOM|cOM|COM)?$'
    )
    if re.match(junk_pattern, clean_s):
        return True
    if re.match(r'(?i)^dis[a-z]{3,}(?:ness|tess|cess|erian|minery)$', clean_s):
        return True
    if re.match(r'(?i)^[a-z]{5,}(?:ness|tess|cess)$', clean_s) and clean_s.lower() not in {"business", "fitness", "wellness"}:
        return True
    # Từ rác standalone không khớp pattern prefix+suffix
    junk_standalone = {'quantousus', 'quantous', 'unstitute', 'discepter', 'collo', 'coliner', 'communs', 'co1', 'derrigermin1',
                       'rorizo', 'pronaganda', 'mazzer', 'disserian', 'chotol', 'priviness', 'toescaly', 'pravered', 'dismitted',
                       'disminery'}
    if clean_s.lower() in junk_standalone:
        return True
    # Từ bị lẫn chữ số vào giữa từ (ví dụ ororiz00ne, col0ner, ene00n)
    if re.search(r'[a-z]\d{2,}[a-z]', clean_s.lower()) and len(clean_s) >= 5:
        return True

    return False


def _clean_junk_words(s: str) -> str:
    words = s.split()
    cleaned_words = []
    for w in words:
        w_clean = re.sub(r'[^a-zA-Z0-9]', '', w)
        if re.search(r'\d{4,}', w_clean) or re.match(r'^0{3,}', w_clean):
            continue
        junk_pattern = (
            r'(?i)^(con|col|com|comm|cor|cot|cos|can|trn|trans|contra|contr|inter|hype|como|trns|colinter|disterat|terat|ant|antr|dis|disc|discol)'
            r'(s|es|ers|ins|inters|ess|he|ste|cars|c|ar|shone|gers|tracess|ces|cess|monums|phousness|anshone|cars|ication|che|tess|ness|mogers'
            r'|eritonerizede|anaTis|naTis|interpLat|interpOUm|prOUm|OUm|com|intersm|intersM|vers|sstering|teris|tracOM|cOM|COM)+$'
        )
        if re.match(junk_pattern, w_clean):
            continue
        junk_standalone = {'quantousus', 'quantous', 'unstitute', 'discepter', 'collo', 'coliner', 'communs', 'co1', 'derrigermin1',
                           'disserian', 'pravered', 'priviness', 'toescaly', 'chotol', 'dismitted',
                           'rorizo', 'pronaganda', 'mazzer', 'disminery'}
        if w_clean.lower() in junk_standalone:
            continue
        # Từ bị lẫn chữ số vào giữa (ví dụ ororiz00ne) - đặc điểm ảo giác OCR
        if re.search(r'[a-z]\d{2,}[a-z]', w_clean.lower()) and len(w_clean) >= 5:
            continue
        cleaned_words.append(w)

    res = " ".join(cleaned_words).strip()
    res = re.sub(r'^[\s,\-/]+|[\s,\-/]+$', '', res).strip()
    return res


def _looks_like_junk_token(token: str, *, is_edge: bool = False) -> bool:
    """Nhận diện token OCR vô nghĩa để không xuất ra kết quả cuối."""
    raw = (token or "").strip()
    clean = re.sub(r'[^A-Za-zÀ-ỹĐđ0-9]', '', raw)
    if not clean:
        return True

    if _is_known_token(clean):
        return False

    key = _strip_vietnamese_accents(clean).lower()
    if key in {"tp", "cn", "k", "q", "p"}:
        return False
    if any(ch.isdigit() for ch in clean):
        return False
    if re.search(r'[À-ỹĐđ]', clean):
        return False
    if clean.isupper() and 2 <= len(clean) <= 6:
        return False
    if is_edge and len(clean) <= 1 and not clean[:1].isupper():
        return True

    artifact_suffixes = (
        "erian", "iness", "teris", "cess", "tess", "tracess", "oum",
        "natis", "shone", "mogers", "phousness", "interpoum",
    )
    if len(key) >= 7 and key.endswith(artifact_suffixes):
        return True
    if key in {"disserian", "quantousus", "unstitute", "discepter", "pravered", "priviness", "toescaly", "chotol", "dismitted",
               "rorizo", "pronaganda", "mazzer", "bete", "luwcate"}:
        return True
    if len(key) >= 8 and not clean[:1].isupper():
        vowels = sum(ch in "aeiouy" for ch in key)
        letters = sum(ch.isalpha() for ch in key)
        if letters and vowels / letters < 0.25:
            return True
    return False


def _drop_stray_leading_edge_token(part: str) -> str:
    """Drop a short leading edge artifact when the remaining text is a strong POI name."""
    words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', part or "")
    if len(words) < 3:
        return part
    first = words[0]
    first_key = _strip_vietnamese_accents(first).lower()
    if any(ch.isdigit() for ch in first) and not any(ch.isalpha() for ch in first):
        return part
    rest = words[1:]
    rest_has_vietnamese = any(re.search(r'[À-ỹĐđ]', w) for w in rest)
    rest_title_or_upper = sum(1 for w in rest if w[:1].isupper() or w.isupper())
    # Drop longer unknown ASCII-like artifact before strong Vietnamese name text.
    # Covers OCR icon hallucinations like `Dirrom Tiệm Nhà...` without hardcoding token.
    first_is_pronounceable_title = (
        first[:1].isupper()
        and first[1:].islower()
        and len(first_key) >= 4
        and (sum(ch in "aeiouy" for ch in first_key) / max(1, sum(ch.isalpha() for ch in first_key))) >= 0.35
    )
    first_artifact_like = (
        first[:1].islower()
        or bool(re.search(r'[a-z][A-Z]{1,}', first))
        or first_key.startswith(('dir', 'dis', 'dist', 'com', 'con', 'col', 'cor', 'pr', 'trn', 'inter'))
        or first_key.endswith(('ness', 'cess', 'tess', 'oum', 'natis', 'shone', 'erian'))
    )
    if (
        len(first_key) > 2
        and not re.search(r'[À-ỹĐđ]', first)
        and not _is_known_token(first)
        and not (first.isupper() and len(first_key) > 1)
        and not first_is_pronounceable_title
        and first_artifact_like
        and rest_has_vietnamese
        and len(rest) >= 2
        and (
            rest_title_or_upper / max(1, len(rest)) >= 0.5
            or (len(rest) >= 4 and first_key.startswith(('dir', 'dis', 'dist', 'pr', 'col', 'com', 'con')))
        )
    ):
        return re.sub(r'^\s*' + re.escape(first) + r'\b\s*', '', part, count=1).strip()
    if len(first_key) > 2:
        return part
    if re.search(r'[À-ỹĐđ]', first) and _is_known_token(first):
        return part
    if first.isupper() and len(first_key) > 1:
        return part
    rest = words[1:]
    title_or_upper = sum(1 for w in rest if w[:1].isupper() or w.isupper())
    has_brand_signal = any(w.isupper() and len(w) >= 2 for w in rest) or title_or_upper / max(1, len(rest)) >= 0.65
    if not has_brand_signal:
        return part
    # Keep known administrative abbreviations; drop unknown short fragments like `Bì` before a brand/name.
    if first_key in {"tp", "q", "p"}:
        return part
    if _is_known_token(first) and not re.search(r'[À-ỹĐđ]', first):
        return part
    return re.sub(r'^\s*' + re.escape(first) + r'\b\s*', '', part, count=1).strip()


def _is_category_or_description_segment(segment: str) -> bool:
    """Nhận diện segment mô tả/category Google Maps, không phải tên chính."""
    words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', segment or "")
    if len(words) > 5:
        return False
    if any(any(ch.isdigit() for ch in w) for w in words):
        return False
    if any(re.search(r'[À-ỹĐđ]', w) for w in words):
        return False
    lower_words = [_strip_vietnamese_accents(w).lower() for w in words]
    generic_category_heads = {
        "store", "shop", "accessories", "accessory", "fashion", "souvenir",
        "restaurant", "cafe", "coffee", "bar", "lounge", "spa", "clinic",
        "office", "department", "market", "mall", "school", "bank", "hotel",
        "attraction", "tourist", "vietnamese", "food", "meal", "takeaway",
        "pharmacy", "drugstore", "hospital", "station", "terminal", "airport",
        "museum", "gallery", "library", "parking", "lot",
    }
    if not (set(lower_words) & generic_category_heads):
        return False
    title_or_upper = sum(1 for w in words if w[:1].isupper() or w.isupper())
    # Category labels are usually plain English words, not mixed brand/title names.
    return title_or_upper <= 1
