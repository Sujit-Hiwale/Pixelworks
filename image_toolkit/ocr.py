"""Optional local Tesseract OCR helpers."""

from __future__ import annotations

from PIL import Image, ImageOps, ImageEnhance
from image_toolkit.processing import MAX_PIXELS


def extract_text(image: Image.Image, *, language: str = "eng", grayscale: bool = True,
                 threshold: bool = False, resize_factor: float = 1.0, deskew: bool = False):
    try:
        import pytesseract
        from pytesseract import TesseractNotFoundError
    except ImportError as exc:
        raise RuntimeError("OCR requires the optional pytesseract Python package.") from exc
    prepared = image.copy()
    if resize_factor < 0.5 or resize_factor > 3:
        raise ValueError("OCR resize factor must be between 0.5× and 3×.")
    if grayscale:
        prepared = ImageOps.grayscale(prepared)
        prepared = ImageEnhance.Contrast(prepared).enhance(1.25)
    if resize_factor != 1:
        new_size = (int(prepared.width * resize_factor), int(prepared.height * resize_factor))
        if new_size[0] * new_size[1] > MAX_PIXELS:
            raise ValueError("Resized OCR image would exceed the 30 megapixel limit.")
        prepared = prepared.resize(new_size, Image.Resampling.LANCZOS)
    if deskew:
        prepared = prepared.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
    if threshold:
        prepared = ImageOps.grayscale(prepared).point(lambda px: 255 if px > 165 else 0)
    deskew_scale = resize_factor
    if deskew:
        try:
            import cv2
            import numpy as np
            arr = np.asarray(prepared.convert("L"))
            binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            coords = cv2.findNonZero(binary)
            if coords is not None and len(coords) > 20:
                angle = cv2.minAreaRect(coords)[-1]
                angle = -(90 + angle) if angle < -45 else -angle
                if abs(angle) < 15:
                    matrix = cv2.getRotationMatrix2D((prepared.width/2, prepared.height/2), angle, 1.0)
                    arr = cv2.warpAffine(np.asarray(prepared), matrix, (prepared.width, prepared.height),
                                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    prepared = Image.fromarray(arr)
        except ImportError as exc:
            raise RuntimeError("Deskew requires opencv-python-headless.") from exc
    try:
        data = pytesseract.image_to_data(prepared, lang=language, output_type=pytesseract.Output.DICT)
    except TesseractNotFoundError as exc:
        raise RuntimeError("Tesseract OCR engine is missing. Install the system tesseract-ocr package.") from exc
    except Exception as exc:
        message = str(exc)
        if "Failed loading language" in message or "Error opening data file" in message:
            raise RuntimeError(f"Tesseract language data for '{language}' is unavailable. Install the matching tesseract-ocr language package.") from exc
        raise RuntimeError(f"OCR failed: {message}") from exc
    words, boxes = [], []
    for i, word in enumerate(data.get("text", [])):
        word = word.strip()
        if word:
            words.append(word)
            boxes.append({"text": word, "left": round(data["left"][i]/deskew_scale), "top": round(data["top"][i]/deskew_scale),
                          "width": round(data["width"][i]/deskew_scale), "height": round(data["height"][i]/deskew_scale),
                          "confidence": data["conf"][i]})
    return " ".join(words), boxes
