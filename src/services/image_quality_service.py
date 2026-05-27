import cv2
import numpy as np

def pil_to_bgr(pil_img):
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def gray(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

def blur_score(gray_img):
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

def contrast_score(gray_img):
    return float(gray_img.std())

def brightness_score(gray_img):
    return float(gray_img.mean())

def detect_skew_angle(gray_img):
    edges = cv2.Canny(gray_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=100, maxLineGap=20)
    if lines is None:
        return 0.0
    angles = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 <= angle <= 45:
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0

def compute_quality(gray_img, angle):
    b = blur_score(gray_img)
    c = contrast_score(gray_img)
    br = brightness_score(gray_img)
    blur_n = min(b / 150.0, 1.0)
    contrast_n = min(c / 80.0, 1.0)
    brightness_n = max(0.0, 1.0 - abs(br - 170.0) / 170.0)
    q = max(0.0, min(1.0, 0.45 * blur_n + 0.40 * contrast_n + 0.15 * brightness_n))
    warnings = []
    if abs(angle) > 4:
        warnings.append("high_skew")
    if b < 60:
        warnings.append("low_blur")
    if c < 35:
        warnings.append("low_contrast")
    if br < 70 or br > 235:
        warnings.append("brightness_outlier")
    return {"blur_score": b, "contrast_score": c, "brightness": br, "skew_angle": angle, "quality_score": q, "warnings": warnings}