# DEVELOP BY ALFACEPUNK22
import cv2
import numpy as np

def threshold_polygon_segmentation(image, points):
    final_clean = np.zeros_like(image)
    intermediate = np.zeros_like(image)
    init_roi = np.zeros_like(image)

    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(init_roi, [pts], isClosed=True, color=255, thickness=2)

    # create black mask and fill the polygon with white
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [pts], 255)
    masked_img = cv2.bitwise_and(image, mask)

    # pick bounding box of the polygon
    x, y, w, h = cv2.boundingRect(pts)
    roi_img = masked_img[y:y+h, x:x+w]

    if roi_img.size == 0:
        return init_roi, intermediate, final_clean

    # Otsu Thresholding 
    non_zero_pixels = roi_img[roi_img > 0]

    if len(non_zero_pixels) == 0:
        return init_roi, intermediate, final_clean

    ret, _ = cv2.threshold(non_zero_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    custom_ret = int(ret * 0.75)
    _, thresh = cv2.threshold(roi_img, custom_ret, 255, cv2.THRESH_BINARY)

    pad = 30
    padded_thresh = cv2.copyMakeBorder(thresh, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    kernel_big_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    morph = cv2.morphologyEx(padded_thresh, cv2.MORPH_CLOSE, kernel_big_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_open)

    kernel_small_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_small_close)

    # Blur for smoother edges, then re-threshold
    morph = cv2.GaussianBlur(morph, (15, 15), 0)
    _, morph = cv2.threshold(morph, 127, 255, cv2.THRESH_BINARY)

    # clean up the borders
    morph = morph[pad:-pad, pad:-pad]
    intermediate[y:y+h, x:x+w] = morph

    # Extract contours and draw on final_clean
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        c += np.array([[x, y]]) 
        cv2.drawContours(final_clean, [c], -1, 255, thickness=cv2.FILLED)

    return init_roi, intermediate, final_clean