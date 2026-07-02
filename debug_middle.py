# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'src')
from src.vision.text_cleaning.final_cleanup import _clean_final_ocr_text
from src.vision.text_cleaning.rescue import _merge_missing_middle_tokens
pairs=[('Trạm giữ xe công TNGo - Quận 1','Trạm giữ xe công cộng TNGo - Quận 1'),('Trạm xe đạp TNGo - Hồ Gươm','Trạm xe đạp công cộng TNGo - Hồ Gươm'),('Nhà văn hóa thiếu Nhi Đồng Nai','Nhà văn hóa thiếu nhi Đồng Nai'),('Trạm giữ xe công TNGo - Quận 1','Trạm giữ xe công TNGo - Quận 1 / Parking lot')]
for b,a in pairs:
 print('BASEC',_clean_final_ocr_text(b))
 print('ALTC ',_clean_final_ocr_text(a))
 print('MERG ',_merge_missing_middle_tokens(b,a))
 print()
