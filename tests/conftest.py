import os
import sys

# Đảm bảo pytest ưu tiên nạp các module cục bộ (config, src) thay vì site-packages hệ thống
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
