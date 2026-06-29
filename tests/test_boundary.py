from src.grid import is_point_in_boundary


def test_is_point_in_boundary_strict():
    # Polygon đơn giản 1x1 quanh gốc (0,0)
    # Lục giác/hình vuông [0, 0] -> [10, 0] -> [10, 10] -> [0, 10] -> [0, 0]
    # Coordinates in GeoJSON: [lng, lat]
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [106.6, 10.7],  # Bottom-Left
                [106.8, 10.7],  # Bottom-Right
                [106.8, 10.9],  # Top-Right
                [106.6, 10.9],  # Top-Left
                [106.6, 10.7],  # Close
            ]
        ],
    }

    # Điểm nằm trong
    assert is_point_in_boundary(10.8, 106.7, geometry)

    # Điểm nằm ngoài hẳn
    assert not is_point_in_boundary(10.5, 106.5, geometry)


def test_is_point_in_boundary_buffered():
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [106.6, 10.7],
                [106.8, 10.7],
                [106.8, 10.9],
                [106.6, 10.9],
                [106.6, 10.7],
            ]
        ],
    }

    # Điểm nằm ngoài một chút (vĩ độ 10.6995, cách biên 10.7 khoảng 0.0005 độ (~55 mét))
    # Nạp buffer_meters = 100m -> phải trả về True
    assert is_point_in_boundary(10.6995, 106.7, geometry, buffer_meters=100.0)

    # Nạp buffer_meters = 10m -> phải trả về False
    assert not is_point_in_boundary(10.6995, 106.7, geometry, buffer_meters=10.0)
