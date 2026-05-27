from pathlib import Path
import cv2
from PIL import Image
from src.services.preprocess_service import PreprocessService

img_path = Path(r"data\raw\case_001\khata\doc_90d66c1ea8b8.webp")
#svc = PreprocessService(denoise=False, sharpen=False)
svc = PreprocessService()

img = Image.open(img_path).convert("RGB")
out, quality = svc.enhance(img)

cv2.imwrite("debug_preprocessed_light.png", out)
print(quality)
print("saved: debug_preprocessed_light.png")