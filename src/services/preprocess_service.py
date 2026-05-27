import cv2
import numpy as np
from .image_quality_service import pil_to_bgr, gray, detect_skew_angle, compute_quality

class PreprocessService:
    def __init__(self, max_side: int = 2400, denoise: bool = False, sharpen: bool = False):
        self.max_side = max_side
        self.denoise = denoise
        self.sharpen = sharpen

    def resize_if_needed(self, image_bgr):
        h, w = image_bgr.shape[:2]
        m = max(h, w)
        if m <= self.max_side:
            return image_bgr
        scale = self.max_side / m
        return cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def deskew(self, image_bgr):
        g = gray(image_bgr)
        angle = detect_skew_angle(g)
        if abs(angle) < 0.3:
            return image_bgr, angle
        h, w = image_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return out, angle

    def enhance(self, pil_image):
        out = pil_to_bgr(pil_image)
        out = self.resize_if_needed(out)
        out, angle = self.deskew(out)
        if self.denoise:
            out = cv2.fastNlMeansDenoisingColored(out, None, 8, 8, 7, 21)
        if self.sharpen:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            out = cv2.filter2D(out, -1, kernel)
        quality = compute_quality(gray(out), angle)
        return out, quality