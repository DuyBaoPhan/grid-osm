from src.canonical_matcher import resolve_canonical_name


def test_high_confidence_nearby_match_uses_canonical():
    match = resolve_canonical_name(
        "PASTA CLUB / Not so Italian / Phes",
        "PASTA CLUB / Not so Italian / Phes",
        {"PASTA CLUB / Not so Italian": {"source": "dom", "lat": 10.0, "lng": 106.0}},
        poi_lat=10.0,
        poi_lng=106.0,
    )
    assert match.action == "use_canonical"
    assert match.selected == "PASTA CLUB / Not so Italian"
    assert not match.needs_review


def test_near_match_below_threshold_flags_review_and_keeps_ocr():
    match = resolve_canonical_name(
        "Vườn Trong Phở, gà Định Connection",
        "Vườn Trong Phở, gà Định Connection",
        {"Vườn Trong Phố, Gia Định Connection": {"source": "dom"}},
        min_accept_score=0.99,
        min_review_score=0.70,
    )
    assert match.selected == "Vườn Trong Phở, gà Định Connection"
    assert match.needs_review
    assert match.reason == "nearby_match_below_accept_threshold"


def test_brand_sensitive_requires_very_high_confidence():
    match = resolve_canonical_name(
        "Highlands Coffee Saigon Post Office",
        "Highlands Coffee Saigon Post Office",
        {"Highlands Coffee Sài Gòn Office": {"source": "osm"}},
        min_accept_score=0.80,
    )
    assert match.selected == "Highlands Coffee Saigon Post Office"
    assert not match.needs_review
    assert match.reason == "clean_ocr_no_nearby_match"


def test_no_nearby_match_keeps_clean_ocr_without_review():
    match = resolve_canonical_name("Cafe Sách Phương Nam", "Cafe Sách Phương Nam", {})
    assert match.selected == "Cafe Sách Phương Nam"
    assert match.action == "keep_ocr"
    assert not match.needs_review
