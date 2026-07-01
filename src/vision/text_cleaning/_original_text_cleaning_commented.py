# Original text_cleaning.py archived before package split.
# Kept as comments only; runtime code lives in split modules.
# # =============================================================
# # text_cleaning.py — OCR text cleanup helpers extracted from original vision.py
# # =============================================================
#
# import logging
# import re
# from typing import List
#
# logger = logging.getLogger(__name__)
#
# def _strip_vietnamese_accents(s: str) -> str:
#     """Loại bỏ hoàn toàn dấu tiếng Việt và đưa về chữ thường (Dùng cho deduplicate)."""
#     s = s.lower()
#     s = re.sub(r"[àáảãạăằắẳẵặâầấẩẫậ]", "a", s)
#     s = re.sub(r"[èéẻẽẹêềếểễệ]", "e", s)
#     s = re.sub(r"[ìíỉĩị]", "i", s)
#     s = re.sub(r"[òóỏõọôồốổỗộơờớởỡợ]", "o", s)
#     s = re.sub(r"[ùúủũụưừứửữự]", "u", s)
#     s = re.sub(r"[ỳýỷỹỵ]", "y", s)
#     s = re.sub(r"[đ]", "d", s)
#     return s
#
#
# def _normalize_vietnamese_place_phrases(text: str) -> str:
#     """Chuẩn hóa cụm địa danh Việt bằng gazetteer ngoài file và fuzzy guard."""
#     try:
#         from src.vietnam_places import normalize_place_phrases
#     except ImportError:
#         try:
#             from vietnam_places import normalize_place_phrases
#         except ImportError:
#             return text
#     return normalize_place_phrases(text)
#
#
# def _normalize_ocr_spelling_by_dictionary(text: str) -> str:
#     """Sửa lỗi OCR theo dictionary/context ngoài vision.py, tránh hardcode POI cụ thể."""
#     try:
#         from src.vietnam_places import normalize_ocr_spelling
#     except ImportError:
#         try:
#             from vietnam_places import normalize_ocr_spelling
#         except ImportError:
#             return text
#     return normalize_ocr_spelling(text)
#
#
# def _is_known_token(token: str) -> bool:
#     """True nếu token là từ tiếng Việt / địa danh có trong từ điển."""
#     try:
#         from src.vietnam_places import is_known_token
#     except ImportError:
#         try:
#             from vietnam_places import is_known_token
#         except ImportError:
#             return False
#     return is_known_token(token)
#
#
# def _choose_better_duplicate_token(left: str, right: str) -> str:
#     """Chọn token tốt hơn khi 2 token OCR liền kề cùng base bỏ dấu."""
#     if left == right:
#         return left
#     left_marks = len(re.findall(r'[À-ỹĐđ]', left or ""))
#     right_marks = len(re.findall(r'[À-ỹĐđ]', right or ""))
#     if right_marks > left_marks:
#         return right
#     if left_marks > right_marks:
#         return left
#     if right[:1].isupper() and not left[:1].isupper():
#         return right
#     return left
#
#
# def _remove_adjacent_duplicate_ocr_tokens(text: str) -> str:
#     """Xóa token OCR lặp liền kề bằng base bỏ dấu, không đụng brand ALLCAPS/số."""
#     if not text:
#         return text
#
#     matches = list(re.finditer(r'[A-Za-zÀ-ỹĐđ0-9]+', text))
#     if len(matches) < 2:
#         return text
#
#     replacements = {}
#     remove_indexes = set()
#     prev_key = None
#     prev_token = None
#     prev_idx = None
#     for idx, match in enumerate(matches):
#         token = match.group(0)
#         key = _strip_vietnamese_accents(token).lower()
#         if (
#             prev_token is not None
#             and prev_key == key
#             and len(key) >= 2
#             and (
#                 not (prev_token.isupper() and token.isupper() and len(key) >= 2)
#                 or key in {"tp", "q", "p", "tx", "tt", "cn", "ubnd", "hcm"}
#             )
#             and not any(ch.isdigit() for ch in prev_token + token)
#             and not ((len(prev_token) == 1 or len(token) == 1) and key not in {"q", "p"})
#         ):
#             keep = _choose_better_duplicate_token(prev_token, token)
#             replacements[prev_idx] = keep
#             remove_indexes.add(idx)
#             prev_token = keep
#             continue
#         prev_key = key
#         prev_token = token
#         prev_idx = idx
#
#     if not remove_indexes and not replacements:
#         return text
#
#     out = []
#     last = 0
#     for idx, match in enumerate(matches):
#         if idx in remove_indexes:
#             out.append(text[last:match.start()].rstrip())
#             last = match.end()
#             continue
#         out.append(text[last:match.start()])
#         out.append(replacements.get(idx, match.group(0)))
#         last = match.end()
#     out.append(text[last:])
#     return re.sub(r'\s+', ' ', ''.join(out)).strip()
#
#
# def _fix_latin_brand_ocr_artifacts(text: str) -> str:
#     """Fix generic OCR artifacts in Latin/brand tokens without changing Vietnamese words."""
#     if not text:
#         return text
#
#     # Vietnamese hyphen artifact between word tokens: `Tòa-nhà` -> `Tòa nhà`.
#     text = re.sub(r'(?<=[A-Za-zÀ-ỹĐđ])[-–—](?=[A-Za-zÀ-ỹĐđ])', ' ', text)
#
#     def _fix_crossed_d(match):
#         token = match.group(0)
#         # OCR sometimes reads Latin capital D as Vietnamese Đ in English brand words.
#         # Only change when rest is plain ASCII and token has no other Vietnamese mark.
#         if re.fullmatch(r'Đ[A-Za-z]{3,}', token):
#             return 'D' + token[1:]
#         return token
#
#     text = re.sub(r'\bĐ[A-Za-z]{3,}\b', _fix_crossed_d, text)
#
#     def _fix_stray_leading_capital(match):
#         token = match.group(0)
#         # OCR/icon edge can glue one uppercase letter before a normal TitleCase token:
#         # `LPost`, `LEfora`. Do not touch real camel/acronym brands like `LPBank`.
#         if not re.fullmatch(r'[A-Z][A-Z][a-z]{3,}', token):
#             return token
#         if len(token) >= 3 and token[2].isupper():
#             return token
#         candidate = token[1:]
#         if re.fullmatch(r'[A-Z][a-z]*ola', candidate):
#             candidate = candidate[:-3] + 'ora'
#         key = _strip_vietnamese_accents(candidate).lower()
#         vowels = sum(ch in 'aeiouy' for ch in key)
#         letters = sum(ch.isalpha() for ch in key)
#         if letters >= 4 and vowels / max(1, letters) >= 0.25:
#             return candidate
#         return token
#
#     return re.sub(r'\b[A-Z][A-Z][a-z]{3,}\b', _fix_stray_leading_capital, text)
#
#
# def _clean_spelling(text: str) -> str:
#     # Loại bỏ quote/bracket/backslash rác từ icon/viền crop.
#     # Giữ apostrophe nằm giữa chữ Latin cho brand hợp lệ như Italiani's.
#     text = re.sub(r"[\"`\\\[\]\{\}]", "", text)
#     text = re.sub(r"(?<![A-Za-zÀ-ỹĐđ])'|'(?![A-Za-zÀ-ỹĐđ])", "", text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     text = _fix_latin_brand_ocr_artifacts(text)
#     text = _normalize_vietnamese_place_phrases(text)
#     text = _normalize_ocr_spelling_by_dictionary(text)
#     text = _remove_adjacent_duplicate_ocr_tokens(text)
#     return text
#
#
# def _has_vietnamese_mark(token: str) -> bool:
#     """True nếu token có dấu tiếng Việt, gồm cả chữ đ/Đ."""
#     return bool(re.search(r'[À-ỹĐđ]', token or ""))
#
#
# def _merge_primary_diacritics(primary_text: str, selected_text: str) -> str:
#     """
#     Khi selected/normalized đúng cấu trúc hơn nhưng sai dấu nhẹ, mượn dấu từ Primary.
#     Chỉ thay token nếu:
#     - token bỏ dấu giống nhau
#     - cả Primary và selected đều có dấu Việt
#     - không đổi số/ký tự brand không dấu
#     Ví dụ: Thợ -> Thọ sẽ khôi phục Thợ; Xe không bị đổi thành Xẻ.
#     """
#     primary_tokens = re.findall(r'[A-Za-zÀ-ỹĐđ0-9]+', primary_text or "")
#     selected_tokens = re.findall(r'[A-Za-zÀ-ỹĐđ0-9]+', selected_text or "")
#     if not primary_tokens or len(primary_tokens) != len(selected_tokens):
#         return selected_text
#
#     replacements = []
#     changed = False
#     for p_tok, s_tok in zip(primary_tokens, selected_tokens):
#         if p_tok == s_tok:
#             replacements.append(s_tok)
#             continue
#         if (
#             _strip_vietnamese_accents(p_tok) == _strip_vietnamese_accents(s_tok)
#             and _has_vietnamese_mark(p_tok)
#             and _has_vietnamese_mark(s_tok)
#             and not any(ch.isdigit() for ch in p_tok + s_tok)
#         ):
#             replacements.append(p_tok)
#             changed = True
#         else:
#             replacements.append(s_tok)
#
#     if not changed:
#         return selected_text
#
#     repl_iter = iter(replacements)
#     return re.sub(r'[A-Za-zÀ-ỹĐđ0-9]+', lambda _: next(repl_iter), selected_text)
#
#
# def _vietnamese_tone_preference(token: str) -> int:
#     """Heuristic nhẹ để chọn dấu Việt tự nhiên hơn giữa hai token cùng base."""
#     t = token or ""
#     score = 0
#     # Dấu sắc/huyền/nặng thường ổn định hơn hỏi/ngã trong OCR nhỏ; không ép tuyệt đối.
#     score += 2 * len(re.findall(r'[áéíóúýấắếốớứ]', t, flags=re.IGNORECASE))
#     score += 1 * len(re.findall(r'[àèìòùỳầằềồờừ]', t, flags=re.IGNORECASE))
#     score += 1 * len(re.findall(r'[ạẹịọụỵậặệộợự]', t, flags=re.IGNORECASE))
#     score -= 1 * len(re.findall(r'[ảẻỉỏủỷẩẳểổởử]', t, flags=re.IGNORECASE))
#     score -= 1 * len(re.findall(r'[ãẽĩõũỹẫẵễỗỡữ]', t, flags=re.IGNORECASE))
#     return score
#
#
# def _has_vietnamese_shaped_vowel(token: str) -> bool:
#     """True nếu token có nguyên âm Việt đặc thù dễ bị OCR làm mất: ơ/ư/ă/â/ê/ô."""
#     return bool(re.search(r'[ăằắẳẵặâầấẩẫậêềếểễệôồốổỗộơờớởỡợưừứửữự]', token or "", flags=re.IGNORECASE))
#
#
# def _merge_best_diacritics(primary_text: str, selected_text: str) -> str:
#     """Chọn dấu tốt hơn giữa Primary và selected khi token cùng base bỏ dấu."""
#     primary_tokens = re.findall(r'[A-Za-zÀ-ỹĐđ0-9]+', primary_text or "")
#     selected_tokens = re.findall(r'[A-Za-zÀ-ỹĐđ0-9]+', selected_text or "")
#     if not primary_tokens or len(primary_tokens) != len(selected_tokens):
#         return selected_text
#
#     replacements = []
#     changed = False
#     for p_tok, s_tok in zip(primary_tokens, selected_tokens):
#         if p_tok == s_tok:
#             replacements.append(s_tok)
#             continue
#         if (
#             _strip_vietnamese_accents(p_tok) == _strip_vietnamese_accents(s_tok)
#             and _has_vietnamese_mark(p_tok)
#             and _has_vietnamese_mark(s_tok)
#             and not any(ch.isdigit() for ch in p_tok + s_tok)
#         ):
#             # If selected token is plain ASCII but primary only adds Vietnamese tone,
#             # do not reintroduce accent for brand/English-like names.
#             if re.fullmatch(r'[A-Za-z]+', s_tok) and not _has_vietnamese_shaped_vowel(p_tok):
#                 replacements.append(s_tok)
#             elif _has_vietnamese_shaped_vowel(p_tok) and not _has_vietnamese_shaped_vowel(s_tok):
#                 replacements.append(p_tok)
#                 changed = True
#             elif _vietnamese_tone_preference(p_tok) > _vietnamese_tone_preference(s_tok):
#                 replacements.append(p_tok)
#                 changed = True
#             else:
#                 replacements.append(s_tok)
#         else:
#             replacements.append(s_tok)
#
#     if not changed:
#         return selected_text
#     repl_iter = iter(replacements)
#     return re.sub(r'[A-Za-zÀ-ỹĐđ0-9]+', lambda _: next(repl_iter), selected_text)
#
#
# def _is_junk_line(s: str) -> bool:
#     """Loại bỏ các dòng chữ rác không phải là tên địa điểm (số điện thoại, mã số thuế, đánh giá...)."""
#     s = s.strip()
#     if not s:
#         return True
#
#     # 1. Định dạng rating dạng "4.74 (53)" hoặc "(53)" hoặc "4.7"
#     if re.match(r'^\d+(\.\d+)?\s*\(\d+\)$', s) or re.match(r'^\(\d+\)$', s):
#         return True
#     if re.match(r'^\d+(\.\d+)?$', s):
#         return True
#
#     # Mới: Loại bỏ các dòng rating kèm chữ phụ như "4.6% (35) - Đặt bàn ngay tại Hum"
#     if re.search(r'^\d+(\.\d+)?%\s*\(\d+\)', s) or re.search(r'^\d+(\.\d+)?\s*\(\d+\)\s*-\s*', s):
#         return True
#
#     # Check for consecutive digits on the original string (with spaces/punctuation)
#     # - 5 consecutive digits anywhere (e.g. "00110", "18000000...")
#     # - or 3 consecutive zeros at the start (e.g. "000Trang...")
#     if re.search(r'\d{5,}', s) or re.match(r'^0{3,}', s):
#         return True
#
#     # Remove spaces and common punctuation to analyze characters
#     clean_s = re.sub(r'[^a-zA-Z0-9]', '', s)
#     if not clean_s:
#         return True
#
#     digits = sum(c.isdigit() for c in clean_s)
#     letters = sum(c.isalpha() for c in clean_s)
#     total = len(clean_s)
#
#     # 2. Định dạng chuỗi số dài (mã số thuế, số điện thoại, zip code...)
#     # Nếu dòng chứa nhiều chữ số (trên 65% ký tự và có ít nhất 4 số)
#     # Loại trừ trường hợp địa chỉ (ví dụ: "123 Lê Lợi" có chữ cái chiếm đa số)
#     if total > 0 and digits >= 4 and (digits / total) > 0.65:
#         if letters == 0 or (letters / total) < 0.20:
#             return True
#
#     # 3. Các từ viết tắt nhiễu do quét nhầm viền icon Google Maps (ví dụ: Cons, COLInters, commes...)
#     # Nhận diện các từ rác bắt đầu bằng tiền tố và kết thúc bằng hậu tố đặc trưng của nét tròn
#     junk_pattern = (
#         r'(?i)^(con|col|com|comm|cor|cot|cos|can|trn|trans|contra|contr|inter|hype|como|trns|colinter|ant|antr|dis|disc|discol)'
#         r'(s|es|ers|ins|inters|ess|he|ste|cars|c|ar|shone|gers|tracess|ces|cess|monums|phousness|anshone|cars|ication|che|tess|ness|mogers'
#         r'|anaTis|naTis|interpLat|interpOUm|prOUm|OUm|com|intersm|intersM|vers|sstering|teris|tracOM|cOM|COM)?$'
#     )
#     if re.match(junk_pattern, clean_s):
#         return True
#     # Từ rác standalone không khớp pattern prefix+suffix
#     junk_standalone = {'quantousus', 'quantous', 'unstitute', 'discepter', 'collo', 'coliner', 'communs', 'co1', 'derrigermin1',
#                        'rorizo', 'pronaganda', 'mazzer', 'disserian', 'chotol', 'priviness', 'toescaly', 'pravered', 'dismitted',
#                        'disminery'}
#     if clean_s.lower() in junk_standalone:
#         return True
#     # Từ bị lẫn chữ số vào giữa từ (ví dụ ororiz00ne, col0ner, ene00n)
#     if re.search(r'[a-z]\d{2,}[a-z]', clean_s.lower()) and len(clean_s) >= 5:
#         return True
#
#     return False
#
#
# def _clean_junk_words(s: str) -> str:
#     words = s.split()
#     cleaned_words = []
#     for w in words:
#         w_clean = re.sub(r'[^a-zA-Z0-9]', '', w)
#         if re.search(r'\d{4,}', w_clean) or re.match(r'^0{3,}', w_clean):
#             continue
#         junk_pattern = (
#             r'(?i)^(con|col|com|comm|cor|cot|cos|can|trn|trans|contra|contr|inter|hype|como|trns|colinter|disterat|terat|ant|antr|dis|disc|discol)'
#             r'(s|es|ers|ins|inters|ess|he|ste|cars|c|ar|shone|gers|tracess|ces|cess|monums|phousness|anshone|cars|ication|che|tess|ness|mogers'
#             r'|eritonerizede|anaTis|naTis|interpLat|interpOUm|prOUm|OUm|com|intersm|intersM|vers|sstering|teris|tracOM|cOM|COM)+$'
#         )
#         if re.match(junk_pattern, w_clean):
#             continue
#         junk_standalone = {'quantousus', 'quantous', 'unstitute', 'discepter', 'collo', 'coliner', 'communs', 'co1', 'derrigermin1',
#                            'disserian', 'pravered', 'priviness', 'toescaly', 'chotol', 'dismitted',
#                            'rorizo', 'pronaganda', 'mazzer', 'disminery'}
#         if w_clean.lower() in junk_standalone:
#             continue
#         # Từ bị lẫn chữ số vào giữa (ví dụ ororiz00ne) - đặc điểm ảo giác OCR
#         if re.search(r'[a-z]\d{2,}[a-z]', w_clean.lower()) and len(w_clean) >= 5:
#             continue
#         cleaned_words.append(w)
#
#     res = " ".join(cleaned_words).strip()
#     res = re.sub(r'^[\s,\-/]+|[\s,\-/]+$', '', res).strip()
#     return res
#
#
# def _looks_like_junk_token(token: str, *, is_edge: bool = False) -> bool:
#     """Nhận diện token OCR vô nghĩa để không xuất ra kết quả cuối."""
#     raw = (token or "").strip()
#     clean = re.sub(r'[^A-Za-zÀ-ỹĐđ0-9]', '', raw)
#     if not clean:
#         return True
#
#     if _is_known_token(clean):
#         return False
#
#     key = _strip_vietnamese_accents(clean).lower()
#     if key in {"tp", "cn", "k", "q", "p"}:
#         return False
#     if any(ch.isdigit() for ch in clean):
#         return False
#     if re.search(r'[À-ỹĐđ]', clean):
#         return False
#     if clean.isupper() and 2 <= len(clean) <= 6:
#         return False
#     if is_edge and len(clean) <= 1 and not clean[:1].isupper():
#         return True
#
#     artifact_suffixes = (
#         "erian", "iness", "teris", "cess", "tess", "tracess", "oum",
#         "natis", "shone", "mogers", "phousness", "interpoum",
#     )
#     if len(key) >= 7 and key.endswith(artifact_suffixes):
#         return True
#     if key in {"disserian", "quantousus", "unstitute", "discepter", "pravered", "priviness", "toescaly", "chotol", "dismitted",
#                "rorizo", "pronaganda", "mazzer", "bete", "luwcate"}:
#         return True
#     if len(key) >= 8 and not clean[:1].isupper():
#         vowels = sum(ch in "aeiouy" for ch in key)
#         letters = sum(ch.isalpha() for ch in key)
#         if letters and vowels / letters < 0.25:
#             return True
#     return False
#
#
# def _drop_stray_leading_edge_token(part: str) -> str:
#     """Drop a short leading edge artifact when the remaining text is a strong POI name."""
#     words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', part or "")
#     if len(words) < 3:
#         return part
#     first = words[0]
#     first_key = _strip_vietnamese_accents(first).lower()
#     if any(ch.isdigit() for ch in first):
#         return part
#     rest = words[1:]
#     rest_has_vietnamese = any(re.search(r'[À-ỹĐđ]', w) for w in rest)
#     rest_title_or_upper = sum(1 for w in rest if w[:1].isupper() or w.isupper())
#     # Drop longer unknown ASCII-like artifact before strong Vietnamese name text.
#     # Covers OCR icon hallucinations like `Dirrom Tiệm Nhà...` without hardcoding token.
#     first_is_pronounceable_title = (
#         first[:1].isupper()
#         and first[1:].islower()
#         and len(first_key) >= 4
#         and (sum(ch in "aeiouy" for ch in first_key) / max(1, sum(ch.isalpha() for ch in first_key))) >= 0.35
#     )
#     first_artifact_like = (
#         first[:1].islower()
#         or bool(re.search(r'[a-z][A-Z]{1,}', first))
#         or first_key.startswith(('dir', 'dis', 'dist', 'com', 'con', 'col', 'cor', 'pr', 'trn', 'inter'))
#         or first_key.endswith(('ness', 'cess', 'tess', 'oum', 'natis', 'shone', 'erian'))
#     )
#     if (
#         len(first_key) > 2
#         and not re.search(r'[À-ỹĐđ]', first)
#         and not _is_known_token(first)
#         and not (first.isupper() and len(first_key) > 1)
#         and not first_is_pronounceable_title
#         and first_artifact_like
#         and rest_has_vietnamese
#         and len(rest) >= 2
#         and rest_title_or_upper / max(1, len(rest)) >= 0.5
#     ):
#         return re.sub(r'^\s*' + re.escape(first) + r'\b\s*', '', part, count=1).strip()
#     if len(first_key) > 2:
#         return part
#     if re.search(r'[À-ỹĐđ]', first) and _is_known_token(first):
#         return part
#     if first.isupper() and len(first_key) > 1:
#         return part
#     rest = words[1:]
#     title_or_upper = sum(1 for w in rest if w[:1].isupper() or w.isupper())
#     has_brand_signal = any(w.isupper() and len(w) >= 2 for w in rest) or title_or_upper / max(1, len(rest)) >= 0.65
#     if not has_brand_signal:
#         return part
#     # Keep known administrative abbreviations; drop unknown short fragments like `Bì` before a brand/name.
#     if first_key in {"tp", "q", "p"}:
#         return part
#     if _is_known_token(first) and not re.search(r'[À-ỹĐđ]', first):
#         return part
#     return re.sub(r'^\s*' + re.escape(first) + r'\b\s*', '', part, count=1).strip()
#
#
# def _is_category_or_description_segment(segment: str) -> bool:
#     """Nhận diện segment mô tả/category Google Maps, không phải tên chính."""
#     words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', segment or "")
#     if len(words) < 2 or len(words) > 5:
#         return False
#     if any(any(ch.isdigit() for ch in w) for w in words):
#         return False
#     if any(re.search(r'[À-ỹĐđ]', w) for w in words):
#         return False
#     lower_words = [_strip_vietnamese_accents(w).lower() for w in words]
#     generic_category_heads = {
#         "store", "shop", "accessories", "accessory", "fashion", "souvenir",
#         "restaurant", "cafe", "coffee", "bar", "lounge", "spa", "clinic",
#         "office", "department", "market", "mall", "school", "bank", "hotel",
#     }
#     if not (set(lower_words) & generic_category_heads):
#         return False
#     title_or_upper = sum(1 for w in words if w[:1].isupper() or w.isupper())
#     # Category labels are usually plain English words, not mixed brand/title names.
#     return title_or_upper <= 1
#
#
# def _clean_final_ocr_text(text: str) -> str:
#     """Cleanup cuối: không để ký tự/từ rác lọt ra output."""
#     if not text:
#         return ""
#     text = _clean_spelling(text)
#     parts = [p.strip() for p in re.split(r'\s*/\s*', text) if p.strip()]
#     if len(parts) >= 2:
#         strong_parts = [p for p in parts if not _is_category_or_description_segment(p)]
#         if strong_parts:
#             parts = strong_parts
#     cleaned_parts = []
#     for part_idx, part in enumerate(parts):
#         part = re.sub(r'^[^A-Za-zÀ-ỹĐđ0-9&]+|[^A-Za-zÀ-ỹĐđ0-9&.]+$', '', part).strip()
#         part = _drop_stray_leading_edge_token(part)
#         part = _clean_junk_words(part)
#         words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', part)
#         preserve_amp_ellipsis = bool(re.search(r'&\s*\.{2,}\s*$', part))
#         kept_words = []
#         for idx, word in enumerate(words):
#             is_edge = idx == 0 or idx == len(words) - 1
#             if word == "&" and is_edge and preserve_amp_ellipsis:
#                 kept_words.append(word)
#                 continue
#             if _looks_like_junk_token(word, is_edge=is_edge):
#                 continue
#             kept_words.append(word)
#         if kept_words:
#             # Nếu không xóa token nào, giữ nguyên dấu câu hợp lệ trong segment (vd: `Chợ Nga, TP`).
#             if len(kept_words) == len(words):
#                 cleaned_parts.append(part)
#             else:
#                 cleaned_parts.append(" ".join(kept_words))
#     cleaned = " / ".join(cleaned_parts)
#     cleaned = re.sub(
#         r'\b([^/]{1,40}\b(?:TP\.?|Q\.?|P\.?))\s*/\s*((?:Hồ\s+)?Chí\s+Minh|Hồ\s+Chí\s+Minh)\b',
#         r'\1 \2',
#         cleaned,
#         flags=re.IGNORECASE,
#     )
#     cleaned = re.sub(
#         r'\b(TP\.?)\s+Chí\s+Minh\b',
#         r'\1 Hồ Chí Minh',
#         cleaned,
#         flags=re.IGNORECASE,
#     )
#     # Restore missing leading/common descriptor tokens only in strong local context.
#     # These are phrase-shape rules, not POI-name hardcodes.
#     cleaned = re.sub(
#         r'^(Vặt\s*[-–—]\s*Nước\s+Mía\b)',
#         r'Ăn \1',
#         cleaned,
#         flags=re.IGNORECASE,
#     )
#     cleaned = re.sub(
#         r'\b(Shop\s+Thời\s+Trang)\s*/\s*(Coin\s+Store)\b',
#         r'\1 Nữ - \2',
#         cleaned,
#         flags=re.IGNORECASE,
#     )
#     cleaned = re.sub(
#         r'\b(Nhà)\s+Và\s+(?=[A-ZÀ-ỸĐ])',
#         r'\1 ',
#         cleaned,
#     )
#     segs = [p.strip() for p in re.split(r'\s*/\s*', cleaned) if p.strip()]
#     if len(segs) >= 2:
#         first_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', segs[0])
#         following = " ".join(segs[1:])
#         following_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', following)
#         if len(first_words) == 1 and following_words:
#             first = first_words[0]
#             first_key = _strip_vietnamese_accents(first).lower()
#             following_has_vietnamese = bool(re.search(r'[À-ỹĐđ]', following))
#             following_title = sum(1 for w in following_words if w[:1].isupper() or w.isupper())
#             if (
#                 len(first_key) >= 4
#                 and not re.search(r'[À-ỹĐđ]', first)
#                 and not _is_known_token(first)
#                 and following_has_vietnamese
#                 and following_title / max(1, len(following_words)) >= 0.5
#             ):
#                 segs = segs[1:]
#                 cleaned = " / ".join(segs)
#     if len(segs) == 2:
#         left_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', segs[0])
#         right_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', segs[1])
#         if len(left_words) >= 3 and len(right_words) >= 3 and _is_known_token(right_words[0]):
#             second_key = _strip_vietnamese_accents(right_words[1]).lower()
#             if second_key in {"giao", "duong", "duan", "le", "street", "road"}:
#                 suffix = right_words[0]
#                 if left_words[-1][:1].isupper() and suffix[:1].islower():
#                     suffix = suffix[:1].upper() + suffix[1:]
#                 cleaned = f"{segs[0]} {suffix}"
#     cleaned = _clean_ocr_edge_segments(cleaned) if '_clean_ocr_edge_segments' in globals() else cleaned
#     if re.search(r'\b(?:DIY|souvenirs?|gifts?|accessories|crafts?)\b', cleaned, flags=re.IGNORECASE):
#         cleaned = re.sub(r'\s+8\s*$', ' &...', cleaned)
#     return cleaned.strip()
#
#
# def _continuation_tokens_are_valid(words: List[str]) -> bool:
#     """True nếu phần nối thêm đủ giống tên địa điểm, không phải mô tả/rating rác."""
#     if not words or len(words) > 5:
#         return False
#     continuation = " ".join(words)
#     if _is_junk_line(continuation) or _looks_like_bad_ocr(continuation):
#         return False
#     if _junk_token_count(continuation) > 0:
#         return False
#     digit_count = sum(ch.isdigit() for ch in continuation)
#     letter_count = sum(ch.isalpha() for ch in continuation)
#     if digit_count and digit_count / max(1, digit_count + letter_count) > 0.20:
#         return False
#
#     connector_keys = {"giao", "duong", "le", "street", "road", "corner", "nga", "tu", "xoay"}
#     title_or_viet = 0
#     connector_count = 0
#     for word in words:
#         key = _strip_vietnamese_accents(word).lower()
#         if key in connector_keys:
#             connector_count += 1
#             continue
#         if word[:1].isupper() or re.search(r'[À-ỹĐđ]', word) or _is_known_token(word):
#             title_or_viet += 1
#     return title_or_viet >= 1 and (title_or_viet + connector_count) == len(words)
#
#
# def _append_missing_known_suffix(base_text: str, alternate_text: str) -> str:
#     """Cứu phần đuôi bị thiếu nếu alternate OCR chứa overlap + continuation sạch."""
#     base = _clean_final_ocr_text(base_text)
#     alt = _clean_final_ocr_text(_clean_spelling(alternate_text))
#     if not base or not alt or base == alt:
#         return base
#
#     base_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', base)
#     alt_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', alt)
#     if len(base_words) < 2 or len(alt_words) <= len(base_words):
#         return base
#
#     base_keys = [_strip_vietnamese_accents(w).lower() for w in base_words]
#     alt_keys = [_strip_vietnamese_accents(w).lower() for w in alt_words]
#     max_tail = min(4, len(base_keys))
#
#     for tail_len in range(max_tail, 0, -1):
#         tail = base_keys[-tail_len:]
#         for start in range(0, len(alt_keys) - tail_len):
#             if alt_keys[start:start + tail_len] != tail:
#                 continue
#             next_idx = start + tail_len
#             continuation_words = alt_words[next_idx:next_idx + 5]
#             if not continuation_words:
#                 continue
#
#             # Chỉ cứu phần cùng segment hoặc phrase giao lộ/vị trí sạch.
#             same_segment = False
#             for seg in re.split(r'\s*/\s*', alt):
#                 seg_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', seg)
#                 seg_keys = [_strip_vietnamese_accents(w).lower() for w in seg_words]
#                 for seg_start in range(0, len(seg_keys) - tail_len):
#                     if seg_keys[seg_start:seg_start + tail_len] == tail and seg_start + tail_len < len(seg_keys):
#                         same_segment = True
#                         break
#                 if same_segment:
#                     break
#             if not same_segment:
#                 continue
#
#             for n in range(min(5, len(continuation_words)), 0, -1):
#                 candidate_words = continuation_words[:n]
#                 if not _continuation_tokens_are_valid(candidate_words):
#                     continue
#                 merged = f"{base} {' '.join(candidate_words)}"
#                 cleaned = _clean_final_ocr_text(merged)
#                 if _score_ocr_text_quality(cleaned) >= _score_ocr_text_quality(base) - 4:
#                     return cleaned
#
#     return base
#
#
# def _merge_overlapping_ocr_continuation(base_text: str, alternate_text: str) -> str:
#     """Merge OCR variants when one ends with tokens that start the other.
#
#     Example shape: `A B C` + `C D E` -> `A B C D E`.
#     Generic guard keeps only clean continuation segments with reasonable length.
#     """
#     base = _clean_final_ocr_text(base_text)
#     alt = _clean_final_ocr_text(_clean_spelling(alternate_text))
#     if not base or not alt or base == alt:
#         return base
#
#     base_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', base)
#     alt_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', alt)
#     if len(base_words) < 3 or len(alt_words) < 2:
#         return base
#
#     base_keys = [_strip_vietnamese_accents(w).lower() for w in base_words]
#     alt_keys = [_strip_vietnamese_accents(w).lower() for w in alt_words]
#     max_overlap = min(4, len(base_keys), len(alt_keys))
#
#     for overlap in range(max_overlap, 0, -1):
#         if base_keys[-overlap:] != alt_keys[:overlap]:
#             continue
#         continuation_words = alt_words[overlap:]
#         if not continuation_words or len(continuation_words) > 5:
#             return base
#         continuation = " ".join(continuation_words)
#         if _is_junk_line(continuation) or _junk_token_count(continuation) > 0:
#             return base
#         if _looks_like_bad_ocr(continuation):
#             return base
#         digit_count = sum(ch.isdigit() for ch in continuation)
#         letter_count = sum(ch.isalpha() for ch in continuation)
#         if digit_count and digit_count / max(1, digit_count + letter_count) > 0.25:
#             return base
#         merged = f"{base} {continuation}".strip()
#         if _score_ocr_text_quality(merged) >= _score_ocr_text_quality(base) - 4:
#             return _clean_final_ocr_text(merged)
#         return base
#
#     return base
#
#
# def _junk_token_count(text: str) -> int:
#     parts = [p.strip() for p in re.split(r'\s*/\s*', text or "") if p.strip()]
#     count = 0
#     for part_idx, part in enumerate(parts):
#         words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', part)
#         for idx, word in enumerate(words):
#             is_edge = idx == 0 or idx == len(words) - 1
#             if _looks_like_junk_token(word, is_edge=is_edge):
#                 count += 1
#     return count
#
#
# def _looks_like_bad_ocr(text: str) -> bool:
#     """Nhận diện kết quả OCR có khả năng rác để thử fallback PaddleOCR recognition."""
#     s = (text or "").strip()
#     if not s:
#         return True
#     clean = re.sub(r'[^A-Za-zÀ-ỹ0-9]', '', s)
#     if len(clean) <= 4:
#         return True
#     lower = _strip_vietnamese_accents(s).lower()
#     junk_tokens = (
#         "obst", "obs", "overstress", "couth", "quts", "orns", "orng",
#         "ongame", "oriem", "ducas", "seruper", "postotice", "pertume",
#         "obtrined", "parigheness", "qutminh", "qut"
#     )
#     if any(tok in lower for tok in junk_tokens):
#         return True
#     words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', s)
#     if len(words) == 1:
#         token = words[0]
#         key = _strip_vietnamese_accents(token).lower()
#         if not _is_known_token(token) and not re.search(r'[À-ỹĐđ\d]', token):
#             # Icon/edge hallucinations often become one weird Latin token:
#             # mixed camel/allcaps tail (`ComonESSt`) or artifact prefixes (`Disted`).
#             has_internal_upper = bool(re.search(r'[a-z][A-Z]{1,}', token))
#             artifact_prefix = key.startswith((
#                 'com', 'con', 'col', 'cor', 'cos', 'cot', 'dis', 'dist',
#                 'dir', 'der', 'pr', 'trn', 'trans', 'inter', 'contra', 'hype'
#             ))
#             artifact_suffix = key.endswith((
#                 'esst', 'ess', 'sst', 'sted', 'chest', 'ness', 'cess',
#                 'tess', 'tracess', 'oum', 'natis', 'shone', 'ication',
#                 'ification', 'inten', 'inter', 'inters'
#             ))
#             if len(key) >= 6 and (has_internal_upper or (artifact_prefix and artifact_suffix)):
#                 return True
#     if words and len(words) <= 2 and not any(ch.isdigit() for ch in s):
#         vowel_count = sum(ch in 'aeiouyàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ' for ch in lower)
#         letter_count = sum(ch.isalpha() for ch in lower)
#         if letter_count and vowel_count / letter_count < 0.28:
#             return True
#     return False
#
#
# def _looks_like_vietnamese_gibberish(text: str) -> bool:
#     """Detect multi-token Vietnamese-looking OCR hallucination without hardcoding names."""
#     words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', text or "")
#     word_tokens = [w for w in words if w != "&"]
#     if len(word_tokens) < 3:
#         return False
#     if any(any(ch.isdigit() for ch in w) for w in word_tokens):
#         return False
#     if any(w.isupper() and 2 <= len(w) <= 6 for w in word_tokens):
#         return False
#     if not any(re.search(r'[À-ỹĐđ]', w) for w in word_tokens):
#         return False
#
#     known_count = sum(1 for w in word_tokens if _is_known_token(w))
#
#     unknown_marked = 0
#     odd_repeated_sound = 0
#     keys = []
#     for w in word_tokens:
#         key = _strip_vietnamese_accents(w).lower()
#         keys.append(key)
#         if re.search(r'[À-ỹĐđ]', w) and not _is_known_token(w):
#             unknown_marked += 1
#         if len(key) >= 3 and key[:1] in {'n', 't', 'v'} and key.endswith(('an', 'en', 'em', 'ien')):
#             odd_repeated_sound += 1
#
#     title_or_upper = sum(1 for w in word_tokens if w[:1].isupper() or w.isupper())
#     first_unknown_marked = bool(re.search(r'[À-ỹĐđ]', word_tokens[0]) and not _is_known_token(word_tokens[0]))
#     if known_count >= max(2, len(word_tokens) // 2) and unknown_marked < 2 and not (first_unknown_marked and title_or_upper <= 1):
#         return False
#
#     if first_unknown_marked and len(word_tokens) >= 4 and title_or_upper <= 1:
#         return True
#
#     if unknown_marked >= 2 and known_count == 0:
#         return True
#     if unknown_marked >= 2 and odd_repeated_sound >= 2:
#         return True
#     if len(set(keys)) <= len(keys) - 2 and unknown_marked >= 2:
#         return True
#     return False
#
#
# def _score_ocr_text_quality(text: str, fx: float = 1.0) -> float:
#     """Chấm điểm chất lượng OCR tổng quát, không phụ thuộc keyword/tên riêng."""
#     s = (text or "").strip()
#     if not s:
#         return -9999
#
#     words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', s)
#     letters = re.findall(r'[A-Za-zÀ-ỹ]', s)
#     digits = re.findall(r'\d', s)
#     vietnamese_marks = re.findall(r'[À-ỹ]', s)
#     clean_len = len(re.sub(r'[^A-Za-zÀ-ỹ0-9]', '', s))
#
#     score = clean_len
#     score += 8 * max(0, s.count(' / '))
#     score += 4 * max(0, len(words) - 1)
#     score += 2 * len(vietnamese_marks)
#
#     if _looks_like_bad_ocr(s):
#         score -= 80
#     if _looks_like_vietnamese_gibberish(s):
#         score -= 55
#
#     if letters:
#         digit_ratio = len(digits) / max(1, len(letters) + len(digits))
#         if digit_ratio > 0.35:
#             score -= int(60 * digit_ratio)
#
#     artifact_penalty = _ocr_artifact_score(s) if '_ocr_artifact_score' in globals() else 0
#     score -= 14 * artifact_penalty
#     if '_junk_token_count' in globals():
#         score -= 45 * _junk_token_count(s)
#
#     # Token 1 ký tự ở đầu/cuối thường là mẩu icon hoặc chữ rác.
#     if words and len(words[0]) == 1 and len(words) > 1:
#         score -= 18
#     if words and len(words[-1]) == 1 and len(words) > 1:
#         score -= 12
#
#     # Áp dụng cùng rule cho từng segment ngăn bởi '/', vì lỗi thường xuất hiện dạng "D Little...".
#     for segment in re.split(r'\s*/\s*', s):
#         seg_words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', segment)
#         if len(seg_words) > 1 and len(seg_words[0]) == 1:
#             score -= 28
#         if len(seg_words) > 1 and len(seg_words[-1]) == 1:
#             score -= 16
#         # Ending ngắn sau token dài thường là chữ bị cụt/nối dòng sai: "Steakhous / Ste".
#         if len(seg_words) == 1 and len(seg_words[0]) <= 3 and segment == s.split('/')[-1].strip():
#             previous_words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', ' / '.join(re.split(r'\s*/\s*', s)[:-1]))
#             if previous_words and len(previous_words[-1]) >= 7:
#                 score -= 35
#
#     # Dòng quá ngắn chỉ chấp nhận nếu nó là label ngắn thật; cho điểm thấp để variant dài hơn thắng.
#     if clean_len <= 4:
#         score -= 30
#
#     return score + 0.05 * fx
#
#
# def _ocr_artifact_score(text: str) -> int:
#     """Đếm dấu hiệu OCR méo chữ/dấu câu, không phụ thuộc tên riêng."""
#     s = text or ""
#     score = 0
#     score += 3 * len(re.findall(r'[?#*"“”]', s))
#     score += 2 * len(re.findall(r'[():;]', s))
#     score += 2 * len(re.findall(r'(?<=\w)[\-–—](?=\w)', s))  # dấu gạch chen trong token: -laan
#     score += 2 * len(re.findall(r'\d[A-Za-zÀ-ỹ]|[A-Za-zÀ-ỹ]\d', s))  # 5Chạt
#     score += len(re.findall(r"[^\w\sÀ-ỹ/&.,%+'\-–—]", s))
#     for segment in re.split(r'\s*/\s*', s):
#         words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', segment)
#         if len(words) > 1 and len(words[0]) == 1:
#             score += 2
#         if len(words) > 1 and len(words[-1]) == 1:
#             score += 1
#         for word in words:
#             if any(ch.isdigit() for ch in word):
#                 continue
#             viet_marks = len(re.findall(r'[À-ỹĐđ]', word))
#             # Small-map OCR sometimes glues neighboring Vietnamese words into one
#             # over-accented token, e.g. `Chínhồ`. Treat as artifact, not better text.
#             if len(word) >= 6 and viet_marks >= 2 and word[:1].islower():
#                 score += 4
#             if len(word) >= 7 and viet_marks >= 3:
#                 score += 5
#     return score
#
#
# def _normalized_regresses_quality(primary_text: str, norm_text: str) -> bool:
#     """Giữ Primary nếu normalized không cải thiện rõ mà làm méo token/dấu câu/chính tả."""
#     if not primary_text or not norm_text or _looks_like_bad_ocr(primary_text):
#         return False
#
#     primary_artifacts = _ocr_artifact_score(primary_text)
#     norm_artifacts = _ocr_artifact_score(norm_text)
#     primary_tokens = _ocr_tokens(primary_text)
#     norm_tokens = _ocr_tokens(norm_text)
#     if not primary_tokens or not norm_tokens:
#         return False
#
#     overlap = len(set(primary_tokens) & set(norm_tokens))
#     overlap_ratio = overlap / max(1, min(len(primary_tokens), len(norm_tokens)))
#     token_delta = abs(len(norm_tokens) - len(primary_tokens))
#
#     # Normalized cùng nội dung gần như Primary nhưng nhiều artifact hơn: giữ Primary.
#     if overlap_ratio >= 0.70 and norm_artifacts > primary_artifacts:
#         return True
#
#     # Normalized không có artifact hơn, nhưng chỉ là biến thể chính tả/dấu câu của Primary.
#     # Nếu Primary đã tốt, tránh override chỉ khi Normalized thực sự có thêm artifact.
#     if overlap_ratio >= 0.78 and token_delta <= 2 and norm_artifacts > primary_artifacts:
#         return True
#
#     # Normalized thêm segment/từ ngoài khi primary đã đủ dài thường là ăn chữ nhãn cạnh crop.
#     # CHỈ áp dụng khi có sự trùng lặp (overlap) hoặc primary là subsequence của normalized,
#     # để tránh loại bỏ normalized khi primary là rác hoàn toàn khác biệt.
#     if overlap_ratio >= 0.5 or _contains_token_subsequence(norm_tokens, primary_tokens):
#         if len(primary_tokens) >= 4 and len(norm_tokens) > len(primary_tokens) + 1 and norm_artifacts >= primary_artifacts:
#             return True
#
#     admin_acronyms = {"ubnd", "hđnd", "hdnd", "tp", "hcm"}
#     primary_admin = any(tok in admin_acronyms for tok in primary_tokens)
#     norm_admin = any(tok in admin_acronyms for tok in norm_tokens)
#     if primary_admin and not norm_admin and overlap_ratio < 0.75:
#         return True
#
#     return False
#
#
# def _ocr_tokens(text: str) -> List[str]:
#     """Token OCR đã bỏ dấu để so sánh bao hàm, không phụ thuộc tên riêng."""
#     return re.findall(r'[a-z0-9]+', _strip_vietnamese_accents(text or "").lower())
#
#
# def _contains_token_subsequence(container: List[str], needle: List[str]) -> bool:
#     if not needle or len(needle) > len(container):
#         return False
#     for start in range(0, len(container) - len(needle) + 1):
#         if container[start:start + len(needle)] == needle:
#             return True
#     return False
#
#
# def _is_clean_short_brand_candidate(text: str) -> bool:
#     """Primary 1 token brand/acronym sạch: không cho normalized unrelated thay thế."""
#     words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', text or "")
#     if len(words) != 1:
#         return False
#     token = words[0]
#     if not (2 <= len(token) <= 6):
#         return False
#     if not token.isupper():
#         return False
#     if _junk_token_count(token) > 0:
#         return False
#     if _ocr_artifact_score(token) > 0:
#         return False
#     return True
#
#
# def _texts_are_unrelated(left: str, right: str) -> bool:
#     left_tokens = set(_ocr_tokens(left))
#     right_tokens = set(_ocr_tokens(right))
#     if not left_tokens or not right_tokens:
#         return False
#     return left_tokens.isdisjoint(right_tokens)
#
#
# def _normalized_has_valid_main_name_extension(primary_text: str, norm_text: str) -> bool:
#     """True nếu normalized = primary + phần tiếp theo có dạng tên chính, không phải mô tả/category."""
#     primary_tokens = _ocr_tokens(_clean_spelling(primary_text))
#     norm_tokens = _ocr_tokens(_clean_spelling(norm_text))
#     if not primary_tokens or not norm_tokens or len(norm_tokens) <= len(primary_tokens):
#         return False
#
#     match_start = -1
#     for start in range(0, len(norm_tokens) - len(primary_tokens) + 1):
#         if norm_tokens[start:start + len(primary_tokens)] == primary_tokens:
#             match_start = start
#             break
#     if match_start < 0:
#         return False
#
#     extra_leading = norm_tokens[:match_start]
#     extra_trailing = norm_tokens[match_start + len(primary_tokens):]
#     if extra_leading:
#         # Extra phía trước dễ là chữ nhãn khác/icon hơn là tên bị cắt.
#         return False
#     if not extra_trailing:
#         return False
#
#     norm_parts = [p.strip() for p in re.split(r'\s*/\s*', norm_text or "") if p.strip()]
#     primary_key = " ".join(primary_tokens)
#     extra_parts = []
#     seen_primary = False
#     for part in norm_parts:
#         part_tokens = _ocr_tokens(_clean_spelling(part))
#         if not seen_primary and part_tokens and _contains_token_subsequence(part_tokens, primary_tokens):
#             seen_primary = True
#             continue
#         if seen_primary:
#             extra_parts.append(part)
#
#     if not extra_parts:
#         return False
#
#     for part in extra_parts:
#         tokens = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', part)
#         word_tokens = [t for t in tokens if t != "&"]
#         if not word_tokens:
#             return False
#         if _is_junk_line(part) or _looks_like_bad_ocr(part):
#             return False
#
#         meaningful_part = " ".join(word_tokens)
#         if _junk_token_count(meaningful_part) > 0:
#             return False
#
#         # Dòng mô tả/rating/category thường có nhiều dấu câu/số hoặc là câu dài viết thường.
#         # Không dùng keyword riêng theo ngành/tỉnh để tránh hardcode theo trường hợp.
#         if re.search(r'\d+(?:[.,]\d+)?\s*(?:\(|★|\*)', part):
#             return False
#         if re.search(r'\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(?:AM|PM|am|pm)\b', part):
#             return False
#         digit_count = sum(ch.isdigit() for ch in part)
#         letter_count = sum(ch.isalpha() for ch in part)
#         if digit_count and digit_count / max(1, digit_count + letter_count) > 0.18:
#             return False
#
#         word_count = len(word_tokens)
#         has_vietnamese = bool(re.search(r'[À-ỹĐđ]', part))
#         title_or_upper = sum(1 for t in word_tokens if t[:1].isupper() or t.isupper())
#         title_ratio = title_or_upper / max(1, word_count)
#         has_acronym = any(t.isupper() and 2 <= len(t) <= 6 for t in word_tokens)
#         has_name_separator = bool(re.search(r'[&+\-/]', part))
#         mostly_lower = title_or_upper == 0
#
#         if word_count > 7:
#             return False
#         if word_count >= 5 and mostly_lower:
#             return False
#         if word_count >= 4 and not (has_vietnamese or has_acronym or title_ratio >= 0.5 or has_name_separator):
#             return False
#         if not (has_vietnamese or has_acronym or title_ratio >= 0.5 or has_name_separator):
#             return False
#
#     return True
#
#
# def _normalized_adds_suspicious_text(primary_text: str, norm_text: str) -> bool:
#     """
#     Trả True khi normalized chỉ là primary cộng thêm text ngoài mép crop.
#     Rule tổng quát: primary đã nằm nguyên trong normalized, normalized có phần dư ở đầu/cuối,
#     thì coi phần dư là nhiễu trừ khi primary đang rỗng/rác.
#     """
#     primary_tokens = _ocr_tokens(primary_text)
#     norm_tokens = _ocr_tokens(norm_text)
#     if not primary_tokens or not norm_tokens:
#         return False
#     if len(norm_tokens) <= len(primary_tokens):
#         return False
#
#     match_start = -1
#     for start in range(0, len(norm_tokens) - len(primary_tokens) + 1):
#         if norm_tokens[start:start + len(primary_tokens)] == primary_tokens:
#             match_start = start
#             break
#
#     if match_start < 0:
#         return False
#
#     leading_extra = norm_tokens[:match_start]
#     trailing_extra = norm_tokens[match_start + len(primary_tokens):]
#
#     # Primary đã là chuỗi con đầy đủ, normalized chỉ thêm text ở mép crop: giữ primary.
#     # Bắt case "Lightness / Thư viện số..." và mọi nhiễu tương tự, không hardcode.
#     if leading_extra or trailing_extra:
#         return True
#
#     extra_tokens = norm_tokens.copy()
#     for token in primary_tokens:
#         try:
#             extra_tokens.remove(token)
#         except ValueError:
#             pass
#
#     if any(len(tok) <= 1 for tok in extra_tokens):
#         return True
#
#     for segment in re.split(r'\s*/\s*', norm_text or ""):
#         seg_words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', segment)
#         if len(seg_words) > 1 and (len(seg_words[0]) == 1 or len(seg_words[-1]) == 1):
#             return True
#
#     return False
#
#
# def _segment_has_strong_signal(segment: str) -> bool:
#     """Segment có khả năng là tên thật: nhiều từ, có dấu Việt, số địa chỉ, hoặc chữ hoa/thương hiệu ngắn."""
#     words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', segment or "")
#     if not words:
#         return False
#     if len(words) >= 2:
#         return True
#     token = words[0]
#     if re.search(r'[À-ỹ]', token) or re.search(r'\d', token):
#         return True
#     if token.isupper() and 2 <= len(token) <= 6:
#         return True
#     return False
#
#
# def _is_weak_edge_segment(segment: str) -> bool:
#     """Nhận diện segment rìa yếu sinh từ chữ/icon nhãn lân cận, không dựa tên riêng."""
#     s = (segment or "").strip()
#     if re.fullmatch(r'&\s*\.{2,}', s):
#         return False
#     words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', s)
#     if len(words) != 1:
#         return False
#     token = words[0]
#     clean = _strip_vietnamese_accents(token).lower()
#     if len(clean) <= 3:
#         return True
#     if re.search(r'[À-ỹ\d]', token):
#         return False
#     # Lowercase 1 từ ở rìa thường là mảnh chữ nhãn khác: reverses, tybrid...
#     if token[:1].islower() and len(clean) >= 5:
#         return True
#     # Titlecase dài kết thúc bằng đuôi OCR artifact như "Priviness".
#     # Giữ an toàn vì chỉ áp dụng khi token nằm ở mép và phần còn lại có tín hiệu mạnh.
#     if token[:1].isupper() and token[1:].islower() and len(clean) >= 8 and clean.endswith("iness"):
#         return True
#     # ALLCAPS dài không phải acronym ngắn thường là mảnh OCR cạnh crop.
#     if token.isupper() and len(clean) > 6:
#         return True
#     return False
#
#
# def _clean_ocr_edge_segments(text: str) -> str:
#     """Loại segment rác ở đầu/cuối khi kết quả có nhiều segment và lõi đủ mạnh."""
#     parts = [p.strip() for p in re.split(r'\s*/\s*', text or "") if p.strip()]
#     if len(parts) < 2:
#         return (text or "").strip()
#
#     kept = parts[:]
#     while len(kept) >= 2 and _is_weak_edge_segment(kept[0]) and any(_segment_has_strong_signal(p) for p in kept[1:]):
#         kept.pop(0)
#     while len(kept) >= 2 and _is_weak_edge_segment(kept[-1]) and any(_segment_has_strong_signal(p) for p in kept[:-1]):
#         kept.pop()
#
#     return " / ".join(kept)
