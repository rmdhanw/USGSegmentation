import cv2
import pywt
import numpy as np

# Wavelet Denoising
def wavelet_denoising(image):

    coeffs = pywt.wavedec2(image, 'haar', level=2)

    coeffs_H = list(coeffs)

    coeffs_H[1:] = [
        tuple(
            pywt.threshold(subband, value=20, mode='soft')
            for subband in detail
        )
        for detail in coeffs_H[1:]
    ]

    denoised = pywt.waverec2(coeffs_H, 'haar')

    denoised = np.uint8(np.clip(denoised, 0, 255))

    return denoised

# CLAHE Enhancement
def apply_log_filter(image):
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )
    enhanced = clahe.apply(image)

    # use bilateral filter for edge-preserving smoothing
    enhanced = cv2.bilateralFilter(
        enhanced,
        9, 
        75, 
        75
    )

    return enhanced