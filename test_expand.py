import cv2
import numpy as np

img = cv2.imread("screenshots/tile_-1_-1.png")
h_img, w_img = img.shape[:2]

# Detection 6 from YOLO
bbox = [861.0, 983.8, 1050.5, 1025.2]
left, top, right, bottom = bbox

L = max(0, int(left))
R = min(w_img, int(right))
T = max(0, int(top))
B = min(h_img, int(bottom))

# ----------------- ORIGINAL LOGIC -----------------
box_crop = img[T:B, L:R]
hsv_crop = cv2.cvtColor(box_crop, cv2.COLOR_BGR2HSV)
bg_h = np.median(hsv_crop[:, :, 0])
bg_s = np.median(hsv_crop[:, :, 1])
bg_v = np.median(hsv_crop[:, :, 2])
is_white_bg = (bg_s < 25)

new_bottom_orig = B
max_expand = 28

for y in range(B, min(h_img, B + max_expand)):
    row_pixels = img[y, L:R]
    row_hsv = cv2.cvtColor(row_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    
    matching_count = 0
    for i in range(len(row_hsv)):
        h, s, v = row_hsv[i]
        is_bg = False
        if is_white_bg:
            if s < 35 and abs(int(v) - int(bg_v)) < 30:
                is_bg = True
        else:
            dh = min(abs(int(h) - int(bg_h)), 180 - abs(int(h) - int(bg_h)))
            if dh < 15 and abs(int(s) - int(bg_s)) < 40 and abs(int(v) - int(bg_v)) < 40:
                is_bg = True
                
        is_text = False
        if bg_v > 150:
            if v < 110:
                is_text = True
        else:
            if v > 170:
                is_text = True
                
        if is_bg or is_text:
            matching_count += 1
            
    ratio = matching_count / len(row_hsv) if len(row_hsv) > 0 else 0.0
    if ratio > 0.78:
        new_bottom_orig = y
    else:
        print(f"Orig stopped at y={y} because ratio={ratio:.3f} <= 0.78")
        break

print(f"Original new_bottom: {new_bottom_orig}")

# ----------------- IMPROVED COLUMN-BASED LOGIC -----------------
# We compare row pixels with ref pixels at B-1 in same column.
ref_pixels = img[B-1, L:R]
ref_hsv = cv2.cvtColor(ref_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

new_bottom_new = B
for y in range(B, min(h_img, B + max_expand)):
    row_pixels = img[y, L:R]
    row_hsv = cv2.cvtColor(row_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    
    matching_count = 0
    for i in range(len(row_hsv)):
        h, s, v = row_hsv[i]
        ref_h, ref_s, ref_v = ref_hsv[i]
        
        # Local background similarity
        is_bg = False
        # Calculate color distance in BGR
        diff = row_pixels[i].astype(int) - ref_pixels[i].astype(int)
        dist = np.sqrt(np.sum(diff**2))
        if dist < 45:
            is_bg = True
            
        # Local text contrast check
        is_text = False
        if ref_v > 150:
            if v < 110:
                is_text = True
        else:
            if v > 170:
                is_text = True
                
        if is_bg or is_text:
            matching_count += 1
            
    ratio = matching_count / len(row_hsv) if len(row_hsv) > 0 else 0.0
    if ratio > 0.78:
        new_bottom_new = y
    else:
        print(f"New stopped at y={y} because ratio={ratio:.3f} <= 0.78")
        # Let's print some details for the failure row
        break

print(f"New new_bottom: {new_bottom_new}")
