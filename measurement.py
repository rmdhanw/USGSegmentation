import cv2
import numpy as np

def measure_fetus(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    c = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(c)

    x, y, w, h = cv2.boundingRect(c)

    diagonal = np.sqrt(w**2 + h**2)

    return {
        'area_pixel': area,
        'width': w,
        'height': h,
        'diagonal_pixel': diagonal
    }