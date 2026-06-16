import cv2
import numpy as np
import matplotlib.pyplot as plt

from enhancement import wavelet_denoising
from enhancement import apply_log_filter

# PASTIKAN NAMA IMPORT DIUBAH MENJADI FUNGSI POLIGON
from segmentation import threshold_polygon_segmentation
from measurement import measure_fetus

# =====================
# Load Gambar
# =====================
img = cv2.imread('dataset/usg2.png', 0)

if img is None:
    print("Gambar tidak ditemukan!")
    exit()

img = cv2.resize(img, (512, 512))

# =====================
# Enhancement
# =====================
wavelet = wavelet_denoising(img)
enhanced = apply_log_filter(wavelet)

# =====================
# Kalibrasi Poligon ROI (Mouse Click)
# =====================
points = []

def draw_polygon(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

cv2.namedWindow('Kalibrasi ROI')
cv2.setMouseCallback('Kalibrasi ROI', draw_polygon)

print("\n Click C for remove all points, ENTER to finish.")

while True:
    display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if len(points) > 0:
        for i in range(len(points) - 1):
            cv2.line(display_img, points[i], points[i+1], (0, 255, 0), 2)
            cv2.circle(display_img, points[i], 3, (0, 0, 255), -1)
            
        cv2.circle(display_img, points[-1], 3, (0, 0, 255), -1)
        
        if len(points) > 2:
            cv2.line(display_img, points[-1], points[0], (0, 255, 0), 2)
            
    cv2.imshow('Kalibrasi ROI', display_img)
    
    key = cv2.waitKey(1)
    # threshold minim 3 points
    if key == 13 and len(points) > 2:
        break
    elif key == ord('c') or key == ord('C'): 
        points = []

cv2.destroyAllWindows()

# =====================
# Segmentasi Thresholding
# =====================
print("\n[-] Memproses Segmentasi...")

init_roi, intermediate, segmented = threshold_polygon_segmentation(enhanced, points)
segmented = segmented.astype('uint8')

# =====================
# Pengukuran
# =====================
result = measure_fetus(segmented)

print("\nHASIL PENGUKURAN")
print(result)

# =====================
# Simpan Hasil
# =====================
cv2.imwrite('output/enhancement/enhanced.png', enhanced)
cv2.imwrite('output/segmentation/segment.png', segmented)

# =====================
# Visualisasi Tahapan
# =====================
fig, ax = plt.subplots(1, 4, figsize=(16, 5))

ax[0].imshow(img, cmap='gray')
ax[0].set_title('(a) Original')
ax[0].axis('off')

ax[1].imshow(img, cmap='gray')
ax[1].contour(init_roi, [127], colors='red')
ax[1].set_title('(b) Polygonal Bounding Box')
ax[1].axis('off')

ax[2].imshow(img, cmap='gray')
ax[2].contour(intermediate, [127], colors='blue')
ax[2].set_title('(c) ROI Otsu + Morph')
ax[2].axis('off')

ax[3].imshow(img, cmap='gray')
ax[3].contour(segmented, [127], colors='green')
ax[3].set_title('(d) Final Result')
ax[3].axis('off')

plt.tight_layout()
plt.show()